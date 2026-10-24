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

`kv_cache.py` 提供连续缓存与 SDPA 参考路径。`paged_cache.py` 管理分页物理存储、写入与逻辑读回，`block_pool.py` 管理块编号及位置映射。每层共享一份 `PagedKVStorage`，各请求持有独立块表并独占已分配块。`_retire()` 统一回收请求资源。

单请求 BF16 生成默认使用 SDPA Decode，调度器使用 BF16/V1 CUDA Decode。V1 每个线程块处理一个 Query 头，V3 由同一 GQA 组的四个 Query 头共享 K/V 加载。INT8 缓存存储量化 K/V 和 scale，模型权重保持 BF16。扩展绑定在 `csrc/bindings.cpp`，内核在 `csrc/paged_decode.cu`，编译目标为 `sm_86`。

## 调度与 Graph

`scheduler.py` 管理 waiting、prefilling、running 和 finished 状态。每轮准入请求、批量 Decode，并在轮次起点处理取消和到期。单请求 runner 为各层单独创建缓存。

首次 `submit()` 前调用 `Scheduler.enable_graphs(max_tokens)`，在共享 KV 池上捕获固定 B1/B4 的 BF16/V1 Decode Graph。其他批量或超容量使用 eager，Prefill 和资源回收沿用原流程。Graph 与混合前向分开启用。

## 跨轮流水

同步 `Scheduler.step()` 为默认路径。启用 Graph 后、首次提交前调用：

```python
pipeline = scheduler.enable_pipeline(overlap=True)
```

之后通过 `pipeline.advance()` 推进、`pipeline.drain()` 收尾。`overlap=False` 使用相同实现并在当轮等待提交，作为同步对照。有在途结果时统一由 pipeline 驱动。

`async_decode.py` 为每个固定批量分配两个交替槽，保存设备 Token、pinned 元数据与输出，以及采样完成和输出就绪事件。计算流执行输入更新、Graph 和选词；拷贝流等待采样事件后取回 Token。下一轮输入直接使用前槽设备 Token。

```text
首次推进：发射 j → 返回
后续推进：发射 j+1 → 等 j 的输出就绪 → CPU 提交 j → 返回
收尾排空：等待最后一轮 → 提交有效结果 → 回收资源
```

推进内部最多有两轮未提交，接口返回时最多一轮。槽位保留至对应事件完成，设备 Token 消费后再覆盖。`cache.length` 表示已登记执行长度；已产生首 Token 的请求，其 CPU 已提交有效 KV 长度为 `输入长度 + 输出数量 - 1`。

请求身份、容量和生成上限满足条件时执行前瞻。B2/B3、等待准入、身份变化或容量边界先排空，再回退同步调度。EOS、取消或到期后丢弃该请求的在途结果，其他请求照常提交；设备引用结束后统一退休。Prefill 保持同步。

设计参考 DeepInfer 的设备 Token 接续和 current-forward / previous-post 执行方式，来源与时间线见[单步流水报告](BENCHMARK.md#di-02受限单步流水消融f5d5c33)。

## 请求级采样

`sampler.py` 的 `RequestSampler` 保存 temperature、top-k、seed 和随机数缓冲。参数在请求提交时校验：

- temperature 为 0 时使用 argmax；随机模式使用 FP32 正规正数温度。
- 每请求独立 CPU `torch.Generator`，按输出上限 N 生成 FP32 uniform 并上传，GPU 缓冲占 `4N` 字节。
- 缺省 seed 来自系统熵；请求安全退休时释放缓冲。

位置 0 对应 Prefill 首 Token。同步路径按已提交输出数索引，流水发射按 `flight.targets - 1` 索引。随机数按位置读取，前瞻、丢弃和回退沿用这一位置约定。

GPU 选词在模型 Graph 外进行：有限 BF16/FP32 logits 转 FP32，按分数降序、同分 Token ID 升序排列，保留 top-k 后计算稳定 softmax，再用逆 CDF 选词。CDF 尾项修正为 1，处理累积舍入。结果留在设备槽中供下一轮读取。

固定环境下，相同 logits、参数、seed 和位置可复现采样。设计参考 DeepInfer 的[请求参数](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/sampling_params.py)与[采样层](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py)；本项目独立实现请求随机流与逆 CDF 选词。

`sampling_backend="reference"` 保留稳定排序参考；显式 `cuda` 对同 K 的 B1/B4、K40/128 使用 `csrc/sampling.cu`。每2048个分数由256线程处理：线程内排序8个键，全块反复取最大值保留K项，再按各段有序队首归并。键组合FP32有序位与反向ID，同分取小ID，正负零统一，越界尾项不参与候选。随后一个warp融合温度、softmax、CDF与选词；概率为FP32，前缀用double累积再写回FP32。其余情况回参考，全greedy仍用批量argmax。

CUDA后端复用请求uniform及逻辑位置，缓存设备温度；组批和临时分配在模型Graph外、当前CUDA流执行，结果写回原设备槽。算法划分参考 DeepInfer [`process_sampling`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py) 的候选缩减/融合阶段及 [DeepSelect](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18) 的大词表小K场景；实现独立编写，不依赖它们的运行库。候选ID需精确对齐；概率沿用现有容差，CDF边界选词差异由检查入口输出定位。

## 检查与测量

`check_model.py` 提供 CUDA/INT8 模型对照，`runner --mode check` 开启 HF 逐步对照。调度、分块、混合前向、Graph、流水和采样各有对应的 `check_*` 入口。

`bench_external.py` 测量离线整批完成时间，`bench_workload.py` 测量闭环负载与流水消融，`paged_decode.py` 提供算子数值检查及调用基准。
