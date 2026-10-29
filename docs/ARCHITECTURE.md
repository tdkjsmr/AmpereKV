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

`scheduler.py`按FIFO维护waiting、唯一prefilling槽位、running和finished。每轮先在轮首处理取消/到期，再Decode、回收、推进一个Prefill块；末块才选首Token并转入running。预留额度覆盖请求后续增长，不能用当前空闲块数代替。`_retire()`统一释放采样状态、KV和预算。

Graph在首次提交前捕获BF16/V1模型前向，默认B1/B4；开发分支可用`enable_graphs(max_tokens, batches=(1, 2, 3, 4))`增加真实B2/B3。容量须16对齐；其他批量或超容量回退eager。静态输入/输出和元数据由图持有，每步重写位置与各层块表；返回logits会被下一次重放覆盖。

`prefill_chunk_size=0`保持整段，正值每轮推进一块。后续块使用`j ≤ history+i`偏移因果掩码；Prefill不进Graph。混合前向共用投影和MLP，Attention仍按请求独立，与Graph分开启用。配置与接口见[README](../README.MD)，取舍见[动态负载报告](BENCHMARK.md#动态到达负载与分块取舍)。

## 跨轮流水

同步`Scheduler.step()`为默认。首次提交前启用Graph，再调用`scheduler.enable_pipeline(overlap=True)`，之后通过`advance()`推进、`drain()`收尾；`overlap=False`作为设备接续的同步对照。

`async_decode.py`为B1/B4分配两个交替槽，持有设备Token、pinned元数据/输出和CUDA事件。计算流执行输入更新、Graph与选词，拷贝流等待采样事件再取回Token；下一轮输入直接使用前槽设备Token。

```text
首次：发射j → 返回
后续：发射j+1 → 等j的输出就绪 → CPU提交j → 返回
排空：等待最后一轮 → 提交有效结果 → 回收
```

内部最多两轮在途、返回时最多一轮。槽位持有至事件完成，旧设备Token消费后才能覆盖。`cache.length`表示已登记执行长度，CPU已提交有效长度为`P+len(output_ids)-1`。B2/B3、等待准入、身份变化或容量边界先排空，再走同步调度。EOS/取消/到期丢弃对应在途结果，设备引用结束后退休；停止只在轮首生效，Prefill保持同步。

设计参考DeepInfer的设备Token接续及current-forward / previous-post方式，来源与时间线见[单步流水报告](BENCHMARK.md#di-02受限单步流水消融f5d5c33)。

## 请求级采样

`RequestSampler`在提交时校验temperature/top-k/seed。每请求私有CPU Generator产生N个FP32 uniform并上传，占`4N`字节，退休时释放。位置0为Prefill首Token；同步按已提交输出数索引，流水按`flight.targets-1`索引，前瞻丢弃不推进共享随机游标。

选词在模型图外进行：分数降序、同分Token ID升序，选足top-k后稳定softmax；已有FP32概率用FP64累计，除以实际总量后转回FP32 CDF，零概率尾部平坦，末端为1。固定环境、后端、参数和位置可复现；不同后端可在CDF移动分界处选中不同候选。

默认`reference`使用稳定排序。显式`cuda`支持同K的B1/B4、K40/128、FP32/BF16和V≤524288，其他组合回退参考。CUDA将大词表分段缩减为小候选，再融合温度、softmax、CDF和选词；全greedy使用argmax。CUDA与参考共用请求随机流，临时组批/分配在图外、当前流执行。

来源：DeepInfer的[请求参数](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/sampling_params.py)、[采样层](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py)、[候选缩减/融合划分](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py)，以及[DeepSelect的大词表小K场景](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18)。请求随机流、逆CDF和CUDA内核由本项目实现。

检查保留候选ID/分数精确对照，概率rtol=2e-6、atol=1e-7，CDF独立FP64及两侧绝对误差上限1e-6。各自选词须符合各自CDF；跨后端差异只接受uniform落在实际移动分界区间内，不修改生产随机数。模型首次分叉诊断在计时外。

## 结构化输出与检查入口

`structured_output.py`通过可选XGrammar 0.1.18公开API编译Schema；编译结果可复用，每请求独占matcher与CPU/GPU bitmask。真实tokenizer和模型词表共同定义合法编号，补齐编号被排除；图外选词前将非法logits设为负无穷，零合法项直接报错。

支持同步eager/Graph与reference/CUDA采样，提交时拒绝流水、mixed、分块及ignore_eos。`_commit`对实际Token调用一次accept_token，EOS标记完整、生成上限标记截断；`_retire`释放matcher及掩码。普通请求不导入XGrammar。接口参考[DeepInfer结构化输出](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/structure_output)。

`runner --mode check`为HF逐步对照，`check_model`覆盖CUDA/INT8模型；调度、分块、Graph、流水、采样和语法各有`check_*`入口。`bench_external`测离线整批，`bench_workload`测闭环与流水，`bench_arrivals`测预定到达，`profile_arrivals`定位阶段，`paged_decode --benchmark`测算子调用。命令与测试表见[README](../README.MD)，配置和数据集中在[性能报告](BENCHMARK.md)。
