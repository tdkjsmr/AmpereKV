# AmpereKV 架构

AmpereKV 在单张 CUDA GPU 上运行 Qwen3-8B。线性层、Embedding、RMSNorm 和 MLP 使用 PyTorch；分页写入与 Decode Attention 使用 C++/CUDA 扩展。

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

## 模型与缓存

`runner.py` 负责模型加载、生成循环和逐层前向。Prefill 输入 Prompt，写入 RoPE 后的 K 和未旋转的 V；后续 Decode 输入上次选出的 Token，并使用其绝对位置和已有 KV。分块 Prefill 的后续块读取已有历史。

`kv_cache.py` 提供连续缓存与 SDPA 参考路径。`paged_cache.py` 管理分页物理存储、写入与逻辑读回，`block_pool.py` 管理块编号及位置映射。调度器为每层建立一份共享的 `PagedKVStorage`，各请求用独立块表把逻辑位置映射到自己持有的物理块。

单请求 BF16 生成默认使用 SDPA Decode，调度器使用 BF16/V1 CUDA Decode。V1 每个线程块处理一个 Query 头，V3 由同一 GQA 组的四个 Query 头共享 K/V 加载。INT8 缓存存储量化 K/V 和 scale，模型权重保持 BF16。扩展绑定在 `csrc/bindings.cpp`，内核在 `csrc/paged_decode.cu`，编译目标为 `sm_86`。

## 调度与 Graph

`scheduler.py` 按 FIFO 管理等待队列，以单个槽位推进 Prefill，并对运行中的请求合批 Decode。每轮先处理取消和到期，再执行 Decode、回收资源并推进一个 Prefill 块。处理完最后一块后，调度器选出首个 Token，并将仍需生成的请求移入运行队列。准入时预留覆盖请求后续增长的块额度。`_retire()` 统一释放采样状态、KV 和额度。

Graph 在首次提交前捕获 BF16/V1 模型前向。默认捕获 B1，批量上限至少为 4 时默认再捕获 B4。批量上限允许时，可调用 `enable_graphs(max_tokens, batches=(1, 2, 3, 4))` 分别捕获四种实际批量。捕获容量以 16 Token 对齐，超出容量或遇到未捕获的批量时回退 eager。Graph 持有固定地址的输入、输出和元数据缓冲区。每步更新位置和各层块表，下一次重放会覆盖本次返回的 logits。

`prefill_chunk_size=0` 执行整段 Prefill。设为正数时，每轮按该大小推进一个块，后续块使用偏移因果掩码 `j ≤ history+i`。同步调度的 Graph 只捕获 Decode。混合前向合并投影和 MLP，Attention 按请求独立执行。混合前向和 Graph 分开启用。配置见 [README](../README.MD)，实测取舍见[动态负载报告](BENCHMARK.md#动态到达负载与分块取舍)。

## 跨轮流水

同步`Scheduler.step()`为默认。首次提交前启用Graph，再调用`scheduler.enable_pipeline(overlap=True)`，之后通过`advance()`推进、`drain()`收尾；`overlap=False`作为设备接续的同步对照。

`async_decode.py`为B1/B4分配两个交替槽，持有设备Token、pinned元数据/输出和CUDA事件。计算流执行输入更新、Graph与选词，拷贝流等待采样事件再取回Token；下一轮输入直接使用前槽设备Token。

```text
首次：发射j → 返回
后续：发射j+1 → 等j的输出就绪 → CPU提交j → 返回
排空：等待最后一轮 → 提交有效结果 → 回收
```

内部最多两轮在途、返回时最多一轮。槽位持有至事件完成，旧设备Token消费后才能覆盖。`cache.length`表示已登记执行长度，CPU已提交有效长度为`P+len(output_ids)-1`。B2/B3、等待准入、身份变化或容量边界先排空，再走同步调度。EOS/取消/到期丢弃对应在途结果，设备引用结束后退休；停止只在轮首生效，Prefill保持同步。

设计参考DeepInfer的设备Token接续及current-forward / previous-post方式，来源与时间线见[流水报告](BENCHMARK.md#单步流水)。

## 请求级采样

`RequestSampler` 在提交时校验温度、top-k 和种子。随机采样使用每请求独立的 CPU Generator，把 N 个 FP32 均匀随机数上传到 GPU，占 `4N` 字节，随请求资源一起释放。首个输出 Token 使用位置 0。同步路径按已提交输出数取值，流水按 `flight.targets-1` 取值。两条路径都直接按位置索引，因此丢弃前瞻结果时无需调整随机数游标。

选词在模型图外执行。候选按分数降序排列，同分时按 Token ID 升序排列。选出 top-k 后做稳定 softmax，以 FP64 累计 FP32 概率，再除以实际总量并转回 FP32 CDF。零概率尾部保持平坦，CDF 末端为 1。后端之间的数值差异可能移动 CDF 分界，从而改变分界附近的选词。

默认`reference`使用稳定排序。显式`cuda`支持同K的B1/B4、K40/128、FP32/BF16和V≤524288，其他组合回退参考。CUDA将大词表分段缩减为小候选，再融合温度、softmax、CDF和选词；全greedy使用argmax。CUDA与参考共用请求随机流，临时组批/分配在图外、当前流执行。

来源：DeepInfer的[请求参数](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/sampling_params.py)、[采样层](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py)、[候选缩减/融合划分](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py)，以及[DeepSelect的大词表小K场景](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18)。请求随机流、逆CDF和CUDA内核由本项目实现。

## 投机解码

`speculative_decode.py`独立驱动8B目标与0.6B草稿，保持各自层数、权重、RoPE和KV池。轮首KV不含最后一个已提交Token（pending）；草稿用pending连续提议K个候选，目标一次输入pending和候选，返回K+1个预测位置。

先取最长匹配前缀。首次不匹配时追加目标模型的修正 Token，全匹配时追加目标模型的额外 Token，再按 EOS 和剩余预算裁切。两套 KV 保留到本轮最后一个新提交 Token 之前。全接受且需要补齐草稿缓存时，草稿对缺少 KV 的候选实际执行一次前向。最后提交的 Token 成为下一轮 pending。

`speculative_draft.DraftGraph`组合既有B1 Graph，独占28层固定池。GPU argmax在图外同流执行，各候选写独立槽，下一步直接读取设备Token；目标验证后一次取回候选与预测。最多四个起草上传槽和一个补算槽，统一取词完成后再复用；结束等待当前流并归还逻辑块，物理池和图跨请求常驻。reference/cuda后端沿用相同接受与回收逻辑。

## 结构化输出与检查入口

`structured_output.py`通过可选XGrammar 0.1.18公开API编译Schema；编译结果可复用，每请求独占matcher与CPU/GPU bitmask。真实tokenizer和模型词表共同定义合法编号，补齐编号被排除；图外选词前将非法logits设为负无穷，零合法项直接报错。

结构化请求使用整段 Prefill、正常 EOS、同步 eager 或 Graph，以及 reference 或 CUDA 采样。提交入口会拒绝流水、mixed、分块和 `ignore_eos` 组合。`_commit` 对实际提交的 Token 调用一次 `accept_token`。EOS 标记语法完整，达到生成上限标记截断。`_retire` 释放 matcher 和掩码。XGrammar 在结构化请求需要编译语法时加载。接口参考 [DeepInfer 结构化输出](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/structure_output)。

`runner --mode check`为HF逐步对照，`check_model`覆盖CUDA/INT8模型；调度、分块、Graph、流水、采样和语法各有`check_*`入口。`bench_external`测离线整批，`bench_workload`测闭环与流水，`bench_arrivals`测预定到达，`profile_arrivals`定位阶段，`paged_decode --benchmark`测算子调用。命令与测试表见[README](../README.MD)，配置和数据集中在[性能报告](BENCHMARK.md)。
