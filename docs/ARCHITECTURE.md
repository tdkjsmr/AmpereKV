# AmpereKV 架构

本项目固定 Qwen3-8B 的一张 CUDA GPU，不实现通用模型注册、网络服务或跨轮会话状态。线性层、Embedding、RMSNorm 和 MLP 等在 PyTorch 中运行；分页写入与 Decode Attention 的关键路径由 C++/CUDA 扩展提供。

```text
单请求 runner.generate_tokens()       多请求 Scheduler.step()
             │                                │
             └──────── model_forward() / model_forward_batched()
                                   │
                   Embedding → 36 个 Decoder Layer → Final Norm/LM Head
                                   │
                    Q/K/V + RoPE → Attention → 残差/MLP
                                   │
                 Prefill: 多 Token + SDPA + 写 KV
                 Decode:  单 Token + 读已有 KV + 追加 KV
                                   │
               每请求 PagedKVCache/BlockTable → 每层 PagedKVStorage/BlockPool
                                   │
                    ampere_kv._C：分页写入、Decode CUDA 内核
```

`ampere_kv/runner.py` 负责模型加载、单请求生成循环和逐层前向；首轮 Prefill 输入整段 Prompt，之后 Decode 每轮只输入上次选出的 Token。`ampere_kv/kv_cache.py` 提供连续缓存与 SDPA 参考路径。`ampere_kv/paged_cache.py` 管理分页物理 K/V、写入、逻辑读回与请求缓存；`ampere_kv/block_pool.py` 管理物理块编号和逻辑位置到块内位置的映射。

`ampere_kv/scheduler.py` 管理等待、Prefill、运行与完成状态，按轮准入、批量 Decode，并在轮次起点合作式处理取消/到期。它为每层创建一份 `PagedKVStorage`：请求各有自己的 `PagedKVCache` 和块表，共享空闲池及物理张量，但同一已分配块不会被两个请求混写。`_retire()` 是完成后的统一归还点；没有前缀共享或引用计数。单请求 `runner` 则为各层单独创建缓存，不经过调度器。

Prefill 产生首个 Token 的 logits，并写入 RoPE 后的 K、未旋转的 V；分块 Prefill 的后续块可读取已有历史。Decode 使用当前 Token 的绝对位置和先前 KV。单请求 BF16 `generate` 默认用 SDPA Decode；调度器默认 BF16/V1 CUDA Decode，单请求的 CUDA 对照在 `cuda-check`，INT8 独立生成需显式 `--kv-dtype int8`。V1 内核每个线程块处理一个 Query 头；可选 V3 内核由同一 GQA 组的四个 Query 头共享 K/V 加载。

在首次 `submit()` 前调用 `Scheduler.enable_graphs(max_tokens)` 可启用固定 B1/B4 的 BF16/V1 Decode Graph；其他批量或超容量回退 eager，Prefill、块所有权与回收不变。Graph 不支持 INT8/V3 或混合前向。`bench_external --engine ampere --cuda-graph` 是该路径的固定负载测量入口，不是服务开关。

## 可选单步流水

同步 `Scheduler.step()` 保持默认。启用 Graph 后、首次提交前，可调用 `pipeline = scheduler.enable_pipeline(overlap=True)`，之后用 `pipeline.advance()` 推进，并在退出前用 `pipeline.drain()` 交付剩余结果。`overlap=False` 使用同一实现但当轮等待，是设备接续同步控制；不复制两套前向。有在途结果时禁止混用 `step()`。

`async_decode.py` 的每个固定批量持有两个交替槽：设备选词、pinned 元数据与输出、采样完成和输出就绪事件。计算流串行执行输入更新、Graph 和 GPU argmax；拷贝流等待本槽采样事件后取回 Token，再记录输出就绪事件。下一轮输入直接消费前槽设备 Token，不等待其 CPU 提交。

```text
首次推进：发射 j → 返回（暂不交付 j）
后续推进：发射 j+1 → 等 j 的输出就绪事件 → CPU 提交 j → 返回
收尾排空：等最后一轮输出 → 提交或丢弃 → 安全退休
```

这里是 current-forward / previous-post：推进内部短暂有两轮未提交结果，接口返回时最多一轮。槽的 pinned 源及输出要保留到其就绪事件完成；计算流顺序确保设备 Token 先被下一轮消费，再轮换覆盖。`cache.length` 表示已登记执行长度，可领先 CPU 提交；对已产生首 Token 的请求，CPU 已提交的有效 KV 长度为 `输入长度 + 输出数量 - 1`。

固定请求身份、容量和上限允许时才前瞻。B2/B3、等待准入、身份变化或容量边界先排空，再回退原同步调度；没有任意批次 remap。EOS 可能在下一轮已发射后才被 CPU 观察到，后续结果必须丢弃；取消/到期也不得提交已观察停止请求的在途结果。邻座合法结果照常提交，所有在途设备引用结束后才走原 `_retire()` 归还，已知生成上限则提前避免无用前瞻。Prefill 保持同步，范围不含 INT8/V3、mixed 或服务化。

机制借鉴 DeepInfer 固定版本的设备 Token 接续、当前前向/前轮后处理及延迟释放，独立实现固定身份子集；不依赖 DeepInfer、TileLang 或其完整 ZOS 调度。来源和实测见[DI-02 报告](BENCHMARK.md#di-02受限单步流水消融f5d5c33)。

扩展绑定见 `csrc/bindings.cpp`，实现见 `csrc/paged_decode.cu`；Python 调用点在 `runner.py` 与 `paged_cache.py`。INT8 存的是量化 K/V 及各自 scale，权重仍为 BF16。当前 CUDA 扩展按 `setup.py` 编译 `sm_86`，不能把源码中出现其他路径视为跨架构兼容性保证。

检查与测量不属于执行主线：`check_model.py` 提供 CUDA/INT8 模型对照；`check_scheduler.py`、`check_chunked.py`、`check_mixed.py`、`check_graph_scheduler.py` 分别检查调度、分块与停止、混合前向、Graph。HF 逐步对照由 `runner --mode check` 开启，正常生成不执行。`bench_external.py` 提供整批计时、输出核对与可选时间线采集；`paged_decode.py` 提供独立算子数值检查及调用基准。
