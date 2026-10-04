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

扩展绑定见 `csrc/bindings.cpp`，实现见 `csrc/paged_decode.cu`；Python 调用点在 `runner.py` 与 `paged_cache.py`。INT8 存的是量化 K/V 及各自 scale，权重仍为 BF16。当前 CUDA 扩展按 `setup.py` 编译 `sm_86`，不能把源码中出现其他路径视为跨架构兼容性保证。

检查与测量不属于执行主线：`check_model.py` 提供 CUDA/INT8 模型对照；`check_scheduler.py`、`check_chunked.py`、`check_mixed.py`、`check_graph_scheduler.py` 分别检查调度、分块与停止、混合前向、Graph。HF 逐步对照由 `runner --mode check` 开启，正常生成不执行。`bench_external.py` 提供整批计时、输出核对与可选时间线采集；`paged_decode.py` 提供独立算子数值检查及调用基准。
