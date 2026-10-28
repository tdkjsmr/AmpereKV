"""单线程 BF16 引擎：共享分页存储、批量 Decode、每轮至多一个 Prefill Chunk、轮次边界合作式停止。"""

import time

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (PAGED_BLOCK_SIZE,
                              model_forward, model_forward_batched, model_forward_mixed)
from ampere_kv.sampler import RequestSampler, sample_batch

WAITING, PREFILLING, RUNNING, FINISHED = "waiting", "prefilling", "running", "finished"


class Request:
    """一个请求的输入、输出与逐层缓存；Prefill 未结束时不进入 Decode 集合。

    `caches[0].length`是已登记执行长度；同步路径从`output_ids`取下个输入，流水可领先提交。
    采样状态在submit时绑定请求，输出位置不是KV长度；EOS与上限仍只写`stop_reason`。
    """

    def __init__(self, request_id: str, input_ids: torch.Tensor, max_new_tokens: int, budget_blocks: int,
                 ignore_eos: bool = False, deadline: float | None = None):
        self.request_id = request_id
        self.input_ids = input_ids  # [1, P] CUDA int64；提交后不再修改。
        self.max_new_tokens = max_new_tokens
        # 调度器级预留额度，与"已分配块数""有效长度"是三回事，见 Scheduler.submit 的注释。
        self.budget_blocks = budget_blocks
        self.ignore_eos = ignore_eos  # 只用于固定工作量的基线；正常生成仍遇 EOS 停止。
        # 绝对截止时刻取自调用方同一进程的 time.monotonic()；None 表示不限，到期只在轮次起点被看到。
        self.deadline = deadline
        self.status = WAITING
        self.caches: list[PagedKVCache] = []
        self.output_ids: list[int] = []
        self.stop_reason = ""
        self.structured_complete = False


class Scheduler:
    """各层共享一个 PagedKVStorage，请求只持有自己的块表；单线程按 FIFO 推进。

    共享的是空闲块池，不是已经分配给某个请求的块：分配出的物理块仍由所属请求独占，所以不
    需要引用计数、锁或所有权移交。释放只发生在 `_retire` 一处。
    """

    def __init__(self, model, blocks_per_layer: int, *, block_size: int = PAGED_BLOCK_SIZE,
                 device="cuda", max_batch: int = 1, prefill_chunk_size: int = 0, mixed: bool = False):
        # 参数边界只查一次；0 保留整段 Prefill，正常前向不重复检查自身产生的状态。
        if max_batch < 1 or prefill_chunk_size < 0:
            raise ValueError("max_batch 必须为正，prefill_chunk_size 不能为负")
        config = model.config
        self.model = model
        self.block_size = block_size
        self.device = device
        # max_batch=1 使用单请求路径；>1 将一组请求的当前 Token 合成一次前向。
        self.max_batch = max_batch
        self.prefill_chunk_size = prefill_chunk_size
        # mixed 默认关闭：已验证的分离路径不被默默替换，实验侧显式选择才走混合前向。
        self.mixed = mixed
        # 两个计数器只服务一件事：报告开关到底生效了几轮、回退了几轮，防止"没混合也报收益"。
        self.mixed_rounds = 0
        self.fallback_rounds = 0
        self.graphs = None
        self.graph_calls = self.graph_fallbacks = self.graph_captures = 0
        self.decode_pipeline = None
        self._submitted = False
        self.blocks_per_layer = blocks_per_layer
        # 每层一份共享物理张量与块池；请求接入时只新建块表，不再分配或复制张量。
        self.storages = [PagedKVStorage(config.num_key_value_heads, config.head_dim, blocks_per_layer,
                                        block_size, device=device) for _ in model.model.layers]
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.prefilling: Request | None = None
        # 各层需求相同，所以一个计数就是每层已预留的块数之和。
        self.reserved_blocks = 0
        eos = model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def enable_graphs(self, max_tokens: int, *, batches: tuple[int, ...] | None = None) -> None:
        """首次提交前捕获真实批量；省略集合时仍选择B1/B4。"""
        if self._submitted or self.graphs is not None or self.mixed:
            raise ValueError("Graph 只能在首次提交前启用，且不与混合轮同时使用")
        if (max_tokens <= 0 or max_tokens % PAGED_BLOCK_SIZE or self.block_size != PAGED_BLOCK_SIZE
                or self.model.config.head_dim != 128 or not self.storages[0]._key.is_cuda
                or self.storages[0]._key.dtype != torch.bfloat16):
            raise ValueError("Graph 仅支持 CUDA BF16 V1、128维、块16及16对齐的正容量")
        batches = tuple(batches) if batches is not None else ((1, 4) if self.max_batch >= 4 else (1,))
        if not batches or any(type(batch) is not int or not 1 <= batch <= min(self.max_batch, 4) for batch in batches):
            raise ValueError("Graph批量集合须非空，且为1至min(max_batch,4)的整数")
        batches = tuple(sorted(set(batches)))
        if self.blocks_per_layer < max(batches):
            raise ValueError("物理池不足以捕获所需批量")
        from ampere_kv.graph_decode import capture_scheduler_graph
        graphs = {batch: capture_scheduler_graph(self.model, self.storages, batch, max_tokens)
                  for batch in batches}
        self.graphs = graphs
        self.graph_max_tokens = max_tokens
        self.graph_captures = len(graphs)
        if self.reserved_blocks or any(storage._pool.num_free_blocks != self.blocks_per_layer
                                       for storage in self.storages):
            raise AssertionError("捕获临时块未全部归还")

    def enable_pipeline(self, *, overlap=True):
        """返回显式advance/drain驱动；overlap=False为同步控制，不替换原step契约。"""
        if (self._submitted or self.graphs is None or self.mixed or self.max_batch not in (1, 4)
                or any(batch not in (1, 4) for batch in self.graphs) or self.decode_pipeline):
            raise ValueError("流水须在首次提交前启用，仅支持已有Graph的BF16/V1 B1/B4，不能mixed")
        from ampere_kv.async_decode import DecodePipeline
        self.decode_pipeline = DecodePipeline(self, overlap=overlap)
        return self.decode_pipeline

    def submit(self, request_id: str, input_ids: torch.Tensor, max_new_tokens: int,
               ignore_eos: bool = False, deadline: float | None = None, *,
               temperature: float = 0.0, top_k: int = -1, seed: int | None = None,
               sampling_backend: str = "reference", structured=None) -> Request:
        """边界校验集中在提交时；内部模型调用不再重复检查自己产出的形状与设备。

        `deadline` 是同一进程 `time.monotonic()` 上的绝对时刻，None 表示不设限、走原有路径。
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("当前只支持单条非空 Token 序列")
        if input_ids.dtype != torch.long or not input_ids.is_cuda:
            raise ValueError("Token IDs 必须是 CUDA 上的 int64 张量")
        if max_new_tokens <= 0:
            raise ValueError("新 Token 上限必须为正数")
        # 保留既有保守预算；实际最终 KV 长度是 P+N-1，最后选出的 Token 不再写回。
        need = (input_ids.shape[1] + max_new_tokens + self.block_size) // self.block_size
        # 只看当前空闲块会高估容量：多个请求会同时消耗尚未兑现的增长空间，最后在 Decode
        # 中途耗尽块池。所以准入用预留额度判断；预算超过整层的请求在提交边界直接拒绝。
        if need > self.blocks_per_layer:
            raise ValueError(f"单请求预算 {need} 块超过每层容量 {self.blocks_per_layer} 块")
        if self.mixed and temperature != 0:
            raise ValueError("随机采样暂不支持mixed模式")
        if structured is not None:
            if self.decode_pipeline is not None or self.mixed or self.prefill_chunk_size or ignore_eos:
                raise ValueError("结构化请求仅支持整段Prefill、同步eager/Graph及正常EOS")
            if structured.vocab_size != self.model.config.vocab_size or set(structured.stop_token_ids) != self.eos_ids:
                raise ValueError("编译语法的词表与停止Token必须匹配模型")
        request = Request(request_id, input_ids, max_new_tokens, need, ignore_eos, deadline)
        request.sampler = RequestSampler(max_new_tokens, self.model.config.vocab_size, self.device,
                                         temperature=temperature, top_k=top_k, seed=seed,
                                         sampling_backend=sampling_backend,
                                         structured=structured.new_state(self.device) if structured is not None else None)
        self._submitted = True
        self.waiting.append(request)
        return request

    def cancel(self, request: Request) -> None:
        """标记取消；同步路径在下轮起点回收，流水路径先排空在途引用；不打断GPU工作。

        只接受本调度器 `submit()` 返回的 Request——这是调用契约，不为此建全局注册表或逐轮归属
        扫描。已完成对象重复调用无作用；已有停止原因不被覆盖，所以取消优先于同一轮的到期判断。
        """
        if request.status != FINISHED:
            request.stop_reason = request.stop_reason or "已取消"

    def _commit(self, request: Request, token: int) -> None:
        """收下本轮选出的 Token 并判断停止；EOS 与达到上限只写 stop_reason，不加新状态。"""
        structured = request.sampler.structured
        if structured is not None:
            request.structured_complete = structured.accept(token)
        request.output_ids.append(token)
        if request.structured_complete:
            request.stop_reason = "EOS"
        elif len(request.output_ids) >= request.max_new_tokens:
            request.stop_reason = "达到生成上限（结构化截断）" if structured is not None else "达到生成上限"
        elif token in self.eos_ids and not request.ignore_eos:
            request.stop_reason = "EOS"

    def _advance(self, request: Request, *, prefill: bool) -> None:
        """推进一个 Prefill Chunk 或一次逐请求 Decode；中间块只更新 KV。"""
        start = request.caches[0].length
        output_logits = True
        if prefill:
            length = request.input_ids.shape[1]
            end = min(start + (self.prefill_chunk_size or length), length)
            input_ids = request.input_ids[:, start:end]
            output_logits = end == length
        else:
            input_ids = request.input_ids.new_tensor([[request.output_ids[-1]]])
        positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # Prefill 始终走 SDPA；Decode 用默认 V1 内核，与单请求生成路径完全相同。
        logits = model_forward(self.model, input_ids, positions, request.caches,
                               is_prefill=prefill, cuda_decode=not prefill, output_logits=output_logits)
        # 位置按已提交输出计数：Prefill末块为0，中间块不消耗随机数。
        if output_logits:
            self._commit(request, request.sampler.sample(logits[0, 0], len(request.output_ids)).item())

    def _decode_batch(self, group: list[Request], graph=None) -> None:
        """将一组请求的当前 Token 合成一次前向，各请求使用独立块表与位置。"""
        tokens = torch.tensor([[request.output_ids[-1]] for request in group],
                              dtype=torch.long, device=self.device)
        # 位置是每个请求各自的绝对位置，不共享——批量不等于对齐历史长度。
        positions = torch.tensor([[request.caches[0].length] for request in group],
                                 dtype=torch.long, device=self.device)
        # 转置成"每层一组请求缓存"：同一层的 B 个缓存指向同一份共享物理存储与块池。
        layer_caches = [[request.caches[layer] for request in group] for layer in range(len(self.storages))]
        logits = (model_forward_batched(self.model, tokens, positions, layer_caches) if graph is None
                  else graph.step(layer_caches, tokens, positions))
        if graph is not None:
            self.graph_calls += 1
        # 一次同步取回 B 个 Token：这是各请求停止判断本身需要的，不是为检查而同步。
        for request, token in zip(group, sample_batch(logits[:, 0], group).tolist()):
            self._commit(request, token)

    def _graph_for(self, group: list[Request]):
        """只读宿主长度，在登记本步新 Token 之前决定是否重放。"""
        if self.graphs is None:
            return None
        graph = self.graphs.get(len(group))
        if graph is None or any(request.caches[0].length >= self.graph_max_tokens for request in group):
            self.graph_fallbacks += 1
            return None
        return graph

    def _decode_round(self) -> list[int]:
        """按 max_batch 分块推进活动请求，返回每一批实际处理的请求数。

        批量大小在真正调用前向的地方记账，不从轮末的活动队列长度反推——同一轮末尾还会接入
        新请求，用队列长度推算会把没参与这次批量 Decode 的请求也算进批里。
        """
        sizes: list[int] = []
        if self.max_batch == 1:
            for request in self.running:
                graph = self._graph_for([request])
                if graph is None:
                    self._advance(request, prefill=False)
                else:
                    self._decode_batch([request], graph)
                sizes.append(1)
            return sizes
        for offset in range(0, len(self.running), self.max_batch):
            group = self.running[offset:offset + self.max_batch]
            self._decode_batch(group, self._graph_for(group))
            sizes.append(len(group))
        return sizes

    def _mixed_round(self) -> int:
        """混合轮：一组 Decode Token 与一个 Prefill 块合成一次逐 Token 前向，返回本批 Decode 数。

        调用方已保证有 pending 且活动数不超过 `max_batch`。块长与"是不是最后一块"在前向之前
        就定死：中间块不选词，末块才产出新请求的首 Token，所以新请求不会在同一轮又被 Decode。
        本轮需要的 Token 一次取回——Decode 的就绪时间因此晚于整次混合计算，这是实测必须体现的成本。
        """
        group = self.running
        pending = self.prefilling
        decode_ids = torch.tensor([[request.output_ids[-1]] for request in group],
                                  dtype=torch.long, device=self.device)
        # 各请求的绝对位置来自自己的缓存长度，与打包进同一次前向的块 Token 无关。
        decode_positions = torch.tensor([[request.caches[0].length] for request in group],
                                        dtype=torch.long, device=self.device)
        layer_caches = [[request.caches[layer] for request in group] for layer in range(len(self.storages))]
        start = pending.caches[0].length
        length = pending.input_ids.shape[1]
        end = min(start + (self.prefill_chunk_size or length), length)
        prefill_ids = pending.input_ids[:, start:end]
        prefill_positions = torch.arange(start, end, device=self.device).unsqueeze(0)
        final = end == length
        decode_logits, prefill_logits = model_forward_mixed(
            self.model, decode_ids, decode_positions, layer_caches,
            prefill_ids, prefill_positions, pending.caches, output_prefill_logits=final)
        picks = (decode_logits[:, 0] if prefill_logits is None
                 else torch.cat((decode_logits[:, 0], prefill_logits[:, 0])))
        tokens = picks.argmax(dim=-1).tolist()  # 一次同步取回 B 个（末块时 B+1 个）Token
        for request, token in zip(group, tokens):
            self._commit(request, token)
        if final:
            self._commit(pending, tokens[-1])
        return len(group)

    def _admit(self) -> None:
        """接入队首到唯一 Prefill 槽位；预算不足时保留 FIFO 等待。"""
        if not self.waiting or self.reserved_blocks + self.waiting[0].budget_blocks > self.blocks_per_layer:
            return
        request = self.waiting.pop(0)
        self.reserved_blocks += request.budget_blocks
        request.caches = [PagedKVCache(storage) for storage in self.storages]
        request.status = PREFILLING
        self.prefilling = request

    def _stop(self, request: Request, now: float) -> bool:
        """本轮要不要停下它；停就顺手在唯一释放点退休，避免"标记了却还占着额度"的中间态。"""
        expired = request.deadline is not None and request.deadline <= now
        if not (request.stop_reason or expired):
            return False
        request.stop_reason = request.stop_reason or "已超时"
        self._retire(request)
        return True

    def _stop_round(self, now: float) -> list[Request]:
        """用本轮起点读到的那一个时刻处理取消与到期；被停下的请求本轮不再启动新前向。

        退休会把状态改成 FINISHED，所以第二次筛表就是"去掉本轮已停的"，不需要另建集合。
        """
        stopped = [request for request in self.waiting if self._stop(request, now)]
        self.waiting = [request for request in self.waiting if request.status != FINISHED]
        stopped += [request for request in self.running if self._stop(request, now)]
        self.running = [request for request in self.running if request.status != FINISHED]
        if self.prefilling is not None and self._stop(self.prefilling, now):
            stopped.append(self.prefilling)
            self.prefilling = None
        return stopped

    def _retire(self, request: Request) -> None:
        """唯一的释放点：归还全部层的块、按退休前状态扣回额度，并把请求移出活动集合。

        WAITING 从未计入 `reserved_blocks`，所以扣减要看状态而不是"KV 长度是否大于 0"——
        整段 Prefill 的请求在第一次前向之前长度同样是 0，用长度判断会把它误扣一次。
        """
        request.sampler.release()  # 流水仅在设备引用排空后退休，也在这里释放随机缓冲。
        for cache in request.caches:
            cache.release()
        request.caches.clear()  # 丢弃块表引用，请求结束后不可能再被误当成活动请求。
        if request.status != WAITING:
            self.reserved_blocks -= request.budget_blocks
        request.status = FINISHED

    def step(self) -> tuple[list[Request], list[int]]:
        """一次调度轮次：合作式停止检查 → Decode（开关打开时可与一个 Prefill 块混合）→ 回收 → Prefill 推进。

        取消与到期只在轮次起点生效：不打断已经发出的 GPU 工作，也不加 `cuda.synchronize` 去
        模拟硬实时，所以轮内到期的请求可能多拿到本轮 Token——这是明示边界。返回的完成请求
        （本轮被停下的加上正常结束的）每轮只交回一次，调度器自己不留历史。
        """
        if self.decode_pipeline is not None and self.decode_pipeline.pending is not None:
            raise RuntimeError("有在途Decode；请先调用流水drain，不可混用step推进")
        finished = self._stop_round(time.monotonic())
        if self.mixed and self.prefilling is None:
            # 混合模式在前向之前先试一次准入，块才能与本轮 Decode 合成一次前向；预算判断照旧，
            # 本轮腾不出空间就只 Decode、下轮起点再准入，不透支尚未回收的块。
            self._admit()
        mixed = self.mixed and self.prefilling is not None and 0 < len(self.running) <= self.max_batch
        if mixed:
            self.mixed_rounds += 1
            sizes = [self._mixed_round()]
        else:
            if self.mixed and self.prefilling is not None and self.running:
                self.fallback_rounds += 1  # 活动数超过单批上限时，退回分离前向。
            sizes = self._decode_round()
        for request in [r for r in self.running if r.stop_reason]:
            self.running.remove(request)
            self._retire(request)
            finished.append(request)
        if self.prefilling is None:
            self._admit()
        request = self.prefilling
        if request is not None and not mixed:
            self._advance(request, prefill=True)
        # 只有最后一块会选词；空输出意味着还在 Prefill，不得提前送入 Decode。
        # 混合轮里末块已经在 `_mixed_round` 内算完，这里只做同一套收尾，不再推进第二次。
        if request is not None and request.output_ids:
            self.prefilling = None
            if request.stop_reason:
                self._retire(request)
                finished.append(request)
            else:
                request.status = RUNNING
                self.running.append(request)
        return finished, sizes


if __name__ == "__main__":
    # 保留旧命令；验收用例不再混在调度器实现中。
    from ampere_kv.check_scheduler import main

    main()
