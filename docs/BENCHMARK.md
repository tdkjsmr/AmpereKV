# 性能与执行取舍

数据来自单张RTX3090（24GiB），目标为固定Qwen3-8B，BF16权重。AmpereKV环境为Python3.12.3、Torch2.5.1+cu124、CUDA12.4、Transformers4.51.0；vLLM单独使用0.29.0、Torch2.13.0+cu130/CUDA13.0。各节按各自源码、输入与计时口径配对，原始样本、Run ID和初始化成本分别保留。

离线整批计时到输出 ID 回到宿主且资源回收。动态负载以计划到达为延迟起点，分别报告到达窗口内的吞吐与计入排空时间的吞吐。Nsight 时间线用于分析执行阶段。

相同 Token 容量下，BF16 KV 的存储字节数是 INT8 KV（含 scale）的 1.9248 倍，相当于节省约 48.05% 的 KV 存储。

## 投机解码

源码`16c02d5fc51793ee71bae9dfdcfda0ef9f5a8c94`，Run ID为`spec03-b-smoke-01`、`spec03-b-paired-01`，工作区干净。目标Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218、草稿Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca，RTX3090、Torch2.5.1+cu124/CUDA12.4和Transformers4.51.0，B1/BF16/greedy、固定64输出并忽略EOS。原cpp/python输入分别41/39Token，输入IDs与SPEC02一致。

`--compare-graph` 测量六条路径，包括普通 `reference`、reference 草稿 K2/K4、Graph 草稿 K2/K4 和普通 `graph-overlap`。两模型和目标、草稿两套图池在各路径运行期间常驻。Graph 草稿复用常驻草稿池，reference 草稿按请求分配逻辑缓存及物理池，这部分分配计入整请求耗时。

正式每路径预热一次、三次样本，顺序正序、逆序、旋转两位。整请求墙钟含Prefill、宿主ID可用、资源归还和末尾等待；模型加载、图初始化、结果核对与打印在计时外。

| 输入 | 路径 | 三次整请求墙钟（ms，按实际出现顺序） | 中位数（ms） |
|---|---|---|---:|
| cpp | reference | 2540.217, 2522.224, 2513.760 | 2522.224 |
| cpp | spec-k2 | 3378.437, 3409.610, 3338.154 | 3378.437 |
| cpp | spec-k4 | 3667.926, 3666.498, 3633.115 | 3666.498 |
| cpp | spec-graph-k2 | 1597.664, 1603.861, 1629.494 | 1603.861 |
| cpp | spec-graph-k4 | 1430.302, 1430.313, 1429.561 | 1430.302 |
| cpp | graph-overlap | 1518.285, 1519.618, 1517.928 | 1518.285 |
| python | reference | 2508.611, 2518.612, 2514.801 | 2514.801 |
| python | spec-k2 | 3061.869, 3052.783, 3085.561 | 3061.869 |
| python | spec-k4 | 3141.318, 3164.452, 3210.499 | 3164.452 |
| python | spec-graph-k2 | 1441.601, 1457.572, 1437.445 | 1441.601 |
| python | spec-graph-k4 | 1246.038, 1246.357, 1251.604 | 1246.357 |
| python | graph-overlap | 1522.123, 1523.164, 1522.737 | 1522.737 |

按未舍入的中位数计算，Graph 草稿 K2/K4 相对各自 reference 草稿的整请求耗时，cpp 下降 52.53%/60.99%，python 下降 52.92%/60.61%。以普通 `graph-overlap` 为基线，cpp 的 K2 耗时增加 5.64%，K4 下降 5.79%。python 的 K2/K4 耗时下降 5.33%/18.15%。cpp 和 python 的 K4 吞吐分别为 44.746 和 51.350 Token/s，对应普通 Graph 的 1.062 和 1.222 倍。

冒烟12请求、正式48请求，共3840个输出ID；每条输入各路径完整64输出均与reference及SPEC02历史输出一致。目标36层最终KV分别104/102，投机草稿28层也为104/102，所有请求回收。正式与预热的同输入同K在新旧草稿间逐轮事件完全一致：

| 输入/K | 提议/实际接受 | 验证轮 | 目标/草稿前向（含Prefill） | 真实补算 | 每请求草稿图重放 |
|---|---:|---:|---:|---:|---:|
| cpp K2 | 53/36 | 27 | 28/72 | 18 | 71 |
| cpp K4 | 81/42 | 21 | 22/91 | 9 | 90 |
| python K2 | 46/39 | 24 | 25/67 | 20 | 66 |
| python K4 | 68/45 | 18 | 19/79 | 10 | 78 |

同一输入、同一 K 的前向次数、接受轨迹和补算次数一致。表中耗时差对应原生 CUDA 草稿前向、Graph 重放、设备端 Token 接续与常驻草稿池复用这一组合。草稿在前向图外的同一流上执行 GPU argmax，下一步直接读取设备 Token。每轮目标验证后统一取回候选与预测，替代逐 Token 取回。宿主仍负责准备和上传元数据。

正式进程中，两套图各捕获一次，容量均为 112 Token。目标图初始化耗时 510.044 ms，allocated 增加 33869312 字节。草稿图初始化耗时 169.851 ms，allocated 增加 21682688 字节（20.678 MiB）。正式草稿图共重放 1220 次，其中提议 992 步、补算 228 次。普通 Graph 共重放 504 次，Decode 回退 0 次。冒烟测试中，草稿图共重放 322 次，其中提议 268 步、补算 54 次。

独立Graph检查覆盖跨块、截断重放、实际补算和同池跨请求；旧默认短回归通过。多Token验证与逐Token路径可能在BF16近同分处选词不同，已记录该案例。

```bash
python -m ampere_kv.check_speculative_decode --model --draft-backend graph --prompt hi --max-new-tokens 64
python -m ampere_kv.bench_speculative_decode --compare-graph --mode smoke --prompt cpp --output /absolute/path/graph-smoke.json --run-id graph-smoke
python -m ampere_kv.bench_speculative_decode --compare-graph --mode paired --output /absolute/path/graph-paired.json --run-id graph-paired
```

原生 CUDA 草稿消融使用提交 `50485d9f4b5f774e3cc822348704c95d97ba93b8`。在 `spec03-a-paired-01` 的 cpp 输入上，K2/K4 中位数从 3315.257/3578.857 ms 降到 2843.469/3000.273 ms，降幅为 14.23%/16.17%。同期普通 `reference` 为 2483.483 ms。独立的 `spec03-a-stages-01` 用宿主墙钟测量阶段耗时，包含等待。K2/K4 的起草与补算合计区间下降 23.25%/22.52%，目标验证耗时相近。

## 离线整批：Graph与vLLM

被测提交为 `7e6bf9b`。同一 RTX 3090（驱动 `595.71.05`）按 E→G→V→V→G→E 顺序运行：E 为同提交 AmpereKV Graph 关闭，G 为 Graph 开启，V 为 vLLM `0.29.0`；B=1/B=4 均每请求 512 输入、固定 32 输出，预热 1 次、正式测量 5 次。下表记录原始五样本，单位 ms；Run ID 与结果 JSON 的 `run_id` 字段一致。

| B | Run ID | 路径 | 五次整批完成时间（ms） | 中位数（ms） | 极差（ms） |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | v2-b1-e1 | E | 1220.521, 1220.584, 1201.770, 1182.426, 1185.914 | 1201.770 | 38.158 |
| 1 | v2-b1-g1 | G | 920.716, 921.000, 920.558, 921.101, 920.947 | 920.947 | 0.543 |
| 1 | v2-b1-v1 | V | 743.659, 744.182, 745.762, 746.545, 747.164 | 745.762 | 3.505 |
| 1 | v2-b1-v2 | V | 746.759, 743.064, 743.817, 749.385, 746.793 | 746.759 | 6.321 |
| 1 | v2-b1-g2 | G | 917.134, 916.882, 917.724, 918.899, 917.296 | 917.296 | 2.017 |
| 1 | v2-b1-e2 | E | 1253.632, 1199.645, 1184.501, 1190.938, 1186.737 | 1190.938 | 69.131 |
| 4 | v2-b4-e1 | E | 1763.927, 1756.897, 1756.665, 1756.762, 1753.137 | 1756.762 | 10.790 |
| 4 | v2-b4-g1 | G | 1495.819, 1498.411, 1496.233, 1491.974, 1491.296 | 1495.819 | 7.115 |
| 4 | v2-b4-v1 | V | 1113.283, 1111.566, 1113.355, 1114.427, 1113.531 | 1113.355 | 2.861 |
| 4 | v2-b4-v2 | V | 1113.755, 1112.077, 1112.694, 1113.356, 1117.821 | 1113.356 | 5.744 |
| 4 | v2-b4-g2 | G | 1495.021, 1497.243, 1519.941, 1501.357, 1500.485 | 1500.485 | 24.920 |
| 4 | v2-b4-e2 | E | 1758.915, 1793.332, 1762.368, 1766.660, 1763.124 | 1763.124 | 34.417 |

同顺序 E/G 配对的整批中位耗时下降：B=1 为 `23.367%`、`22.977%`，B=4 为 `14.854%`、`14.896%`。G/V 完成时间比为 B=1 `1.2349×`、`1.2284×`，B=4 `1.3435×`、`1.3477×`。反向顺序同模式中位数的最大变化为 B=1 的 E `0.901%`、G `0.396%`、V `0.134%`，B=4 的 E `0.362%`、G `0.312%`、V `<0.001%`，均低于 `5%` 环境排查线。

12 份结果 JSON 的实际 Run ID、提交 `7e6bf9b`、输入文件 SHA256 `3262a3904a3c8dfa6d85b8cd63bdc64647783a0d5cd4604496c5166b6362dd80` 已核对；所选输入 SHA256 为 B=1 `e2d1ac38a473406a5e9a26f46a65ba54941fa9eaba11d09d76bf550b1942e36c`、B=4 `22e0ff4cf8fd7422ecd69d068d732d196bd46fb9fc278e3a4ee1df40faf1291a`。B=1 的 E/G/V 输出摘要均为 `a3f96ba20a0fd92cb9583ccd032739ef160f04a48ed10b357754f9eca33c9a59`，B=4 均为 `0418c37d29436ea551dafe674c64daca429fe48bd5a4cd0dbec97a08d121d059`；另对 E1/G1、E2/G2 和两次 G 的逐请求完整序列作直接比较，B1/B4 均一致。AmpereKV 每个正式样本计时外核对与同进程预热完整序列一致、输出数与块归还。

G 使用 BF16/V1、块16、整段 Prefill、`mixed=False`，捕获容量 544 Token、块表宽34；E 保留原 eager。B=1 每样本 Graph 31 次、回退0、捕获总数1；B=4 每样本 Graph 30 次（B4 28、B1 2）、回退4（B2/B3）、捕获总数2，正式重复未重捕获。捕获初始化墙钟：B1 两个 G Run 为 `500.339/516.043 ms`，B4 为 `726.895/722.910 ms`；对应 `allocated` 状态差为 `+17,382,400/+27,192,320` 字节，`reserved` 状态差为 `+46,137,344/+90,177,536` 字节。初始化耗时及显存状态差单独记录。B4 单批 Nsight 时间线中 30 次 `cudaGraphLaunch` 与 28 次 B4、2 次 B1 Graph 范围对应。

vLLM启动配置：Chunked Prefill（`max_num_batched_tokens=4096`）、FLASH_ATTN/FlashAttention 2、PIECEWISE/FULL Graph、KV 预算 `5.88 GiB`、容量 `42,800 Token`，`VLLM_USE_FLASHINFER_SAMPLER=0`。AmpereKV 使用 PyTorch `2.5.1+cu124` / CUDA `12.4`，vLLM 使用 PyTorch `2.13.0+cu130` / CUDA `13.0`。

计时包含调度、前向、输出取回与回收，初始化在计时外。B1 开启 Graph 时同时切换到批量 B1 前向入口，因此 E/G 衡量的是整条执行路径的变化。

## 内部消融：固定 RoPE 频率复用

同一 RTX 3090、B=1/B=4、512 输入/32 固定输出；旧提交 `a19a66a` 与新提交 `914c026`，每进程预热 1 次、测量 5 次。顺序旧→新→新→旧，以下只在各顺序对内比较；对应输出指纹不变。

| B | 顺序对 | 旧版五样本（ms）/中位数 | 新版五样本（ms）/中位数 | 新版完成时间降低 |
|---:|:---:|:---|:---|---:|
| 1 | 旧→新 | 1378.741, 1382.144, 1382.681, 1385.718, 1386.522 / 1382.681 | 1208.197, 1205.226, 1208.321, 1210.355, 1207.444 / 1208.197 | 12.62% |
| 1 | 新→旧 | 1390.707, 1378.239, 1372.658, 1376.401, 1374.858 / 1376.401 | 1209.952, 1210.862, 1208.245, 1210.002, 1196.194 / 1209.952 | 12.09% |
| 4 | 旧→新 | 1973.174, 2004.251, 1942.045, 1955.716, 1946.537 / 1955.716 | 1764.673, 1762.904, 1766.771, 1766.382, 1763.794 / 1764.673 | 9.77% |
| 4 | 新→旧 | 1995.330, 2087.706, 1988.565, 1981.235, 1986.881 / 1988.565 | 1796.480, 1798.880, 1812.104, 1798.426, 1797.410 / 1798.426 | 9.56% |

另一次 B=4 单批 Nsight Systems 诊断中，旧→新 `cudaMemcpyAsync` 与 `cudaStreamSynchronize` 均为 `4066→2698` 次，Kernel launch 均为 `84590` 次。减少的 1368 次与 38 次前向 × 36 层吻合。

## Graph 元数据合并上传消融（7e6bf9b → 0ac0d95）

比较合并上传前的 `7e6bf9b` 与合并后的 `0ac0d95`。同一 RTX 3090、同一输入文件（SHA256 `3262a3904a3c8dfa6d85b8cd63bdc64647783a0d5cd4604496c5166b6362dd80`），B=1/B=4 各请求 512 输入、固定 32 输出并忽略 EOS。两边依赖和 CUDA 扩展相同，源码从各自 checkout 加载。每个 B 按旧→新→新→旧运行，各进程预热一次、测五次。

| B | Run ID | 版本 | 五次整批完成时间（ms） | 中位数（ms） | 极差（ms） |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | v2-r4-b1-old1 | 旧 | 928.312, 929.069, 929.842, 929.914, 930.159 | 929.842 | 1.847 |
| 1 | v2-r4-b1-new1 | 新 | 876.575, 875.933, 875.985, 875.526, 876.707 | 875.985 | 1.181 |
| 1 | v2-r4-b1-new2 | 新 | 876.477, 878.209, 878.582, 876.540, 878.663 | 878.209 | 2.186 |
| 1 | v2-r4-b1-old2 | 旧 | 926.097, 926.630, 928.100, 926.349, 927.422 | 926.630 | 2.003 |
| 4 | v2-r4-b4-old1 | 旧 | 1506.809, 1506.108, 1506.788, 1505.183, 1503.504 | 1506.108 | 3.305 |
| 4 | v2-r4-b4-new1 | 新 | 1457.137, 1456.331, 1459.280, 1456.456, 1459.467 | 1457.137 | 3.136 |
| 4 | v2-r4-b4-new2 | 新 | 1460.986, 1459.815, 1461.681, 1461.538, 1464.060 | 1461.538 | 4.245 |
| 4 | v2-r4-b4-old2 | 旧 | 1490.375, 1494.716, 1498.376, 1513.366, 1508.525 | 1498.376 | 22.991 |

同顺序配对的新版中位整批耗时降低：B=1 为 `5.79%/5.23%`，B=4 为 `3.25%/2.46%`；反向中位数变化均低于 `5%` 环境排查线。八份 JSON 的提交、GPU、输入指纹、逐请求完整输出及 Graph 计数已核对：B=1 每样本重放31次、回退0、捕获总数1；B=4 重放30次（B1两次、B4二十八次）、回退4、捕获总数2。初始化单独记录：B=4 旧版约 `706/752 ms`，新版约 `764/763 ms`。

单批 B=4 Nsight Systems 诊断中，旧→新 `cudaGraphLaunch` 均30次、`cudaLaunchKernel` 均18326次；`cudaMemcpyAsync` 为 `4918→628` 次，`cudaStreamSynchronize` 为 `2698→568` 次。GPU 侧 H2D 为 `2660→530` 次、总量均约 `1.287 MB`，D2D 为 `2220→60` 次、总量约 `1.151→0.002 MB`。次数差与30个 Graph 步每步减少71次 H2D、72次 D2D的预期吻合。采集中的 B4 Graph NVTX 投影均值为 `27.635→25.267 ms`。新版每步构造一份 CPU 元数据张量并执行一次阻塞 H2D。

## 单步流水

本节测量 **512 输入 / 256 输出** 的三路流水消融，实测源码为 `f5d5c3351ae6d15bcc6ae5590f29558e7aaf8f7f`。

### 配置与计时

- 日期：2026-10-23；同一 RTX 3090（24 GiB），Qwen/Qwen3-8B，Revision `b968826d9c46dd6066d109eabc6255188de91218`；PyTorch `2.5.1+cu124`、CUDA `12.4`。
- 单卡、BF16 权重/KV、V1、块16、整段同步 Prefill、greedy、`ignore_eos=True`、`mixed=False`。B=1/B=4 每请求固定生成256个 Token，最后有效 KV 长度767。
- 同进程复用模型、KV池和固定 B1/B4 Graph；容量768、块表宽48，每层池容量为 B×49 块。三路均先用8输出预热，再按 **A→B→C→C→B→A→A→B→C** 测量，每路3样本，样本间恢复空池。
- 从现成 CUDA 输入提交开始，计入宿主驱动、输入更新、前向、输出取回与提交、最终等待和回收。模型加载、分词、输入准备、建池、捕获、槽初始化和预热在计时外；`drain_ms` 已包含在 `total_s` 中。

| 路径 | CLI | 含义 |
|---|---|---|
| A / original-sync | `--pipeline a`（默认） | 原同步 Graph 调度及宿主选词路径，仍是有效默认和回退路径 |
| B / device-sync | `--pipeline b` | GPU argmax、设备 Token 接续，但当轮等待并提交 |
| C / overlap | `--pipeline c` | 与 B 共用输入、采样和缓冲实现，先发射当前轮，再处理上一轮 |

B/C 共用前向、采样和缓冲实现，C/B 比较推进时机；C/A 比较包含设备 Token 接续在内的整体路径。

### 原始正式样本

Run ID 为 `di02-20261023-abc-b1`、`di02-20261023-abc-b4`。全局执行顺序为 `abccbaabc`，下表按路径列出各自3个样本，单位 ms，保留6位小数。

| B | 路径 | 三次整批完成时间（ms） | 中位数（ms） |
|---:|---|---|---:|
| 1 | A | 6181.726936, 6215.981545, 6221.257931 | 6215.981545 |
| 1 | B | 6194.782661, 6209.213601, 6233.832207 | 6209.213601 |
| 1 | C | 6116.856288, 6130.763475, 6167.877464 | 6130.763475 |
| 4 | A | 7163.532436, 7187.946059, 7188.770421 | 7187.946059 |
| 4 | B | 7176.905183, 7193.594730, 7200.728668 | 7193.594730 |
| 4 | C | 7013.971854, 7020.317450, 7035.290379 | 7020.317450 |

按 `100×(1−median(C)/median(B))` 计算，B1 耗时下降 **1.263%**，B4 下降 **2.409%**；相对 A 分别下降 **1.371% / 2.332%**。逐轮 ABC/CBA 配对的 B−C 耗时差为 B1 `77.926 / 78.450 / 65.955 ms`，B4 `162.933 / 173.277 / 165.438 ms`。

| 每个正式样本 | B1 | B4 |
|---|---:|---:|
| 全路径输出数 / Decode 前向调用数 | 256 / 255 | 1024 / 258 |
| Graph 重放 / eager 回退 | 255 / 0 | 254 / 4（B2/B3） |
| Graph 批量分布 / 捕获总数 | B1×255 / 1 | B1×2+B4×252 / 2 |
| B/C 流水前向 / 设备及已提交 Token | 255 / 255 | 253 / 1009 |

B4 流水外执行 5 次回退 Decode，共 11 个 Token。加上流水内的 1009 个 Decode Token 和 Prefill 产生的 4 个首 Token，总输出为 1024 个 Token。正式样本复用预捕获的图，丢弃数为 0。C 的额外宿主推进用于启动与排空，GPU 前向次数与同步路径一致。同一批量的 A/B/C 输出摘要一致。样本结束后，额度归零、36 层块全部归还，allocated 恢复到起始值。

输出滚动摘要：B1 `675379756b89b785c2c304913e2e0fb5131c50617a636235d4912be2b053d181`；B4 `bd98b0668bef8292920551ddb93c4fe635e7b2155ad1e022bdf1e8efc036c3d3`。`check_async_decode` 另覆盖 EOS、取消/到期、容量和退场回退、有效 KV、地址及块复用。自然 EOS 用例中，C 计算12个 Decode 结果、提交11个、丢弃1个。

C 最后输出后的收尾中位数为 B1/B4 `0.406 / 0.333 ms`，已计入总时间。KV 池、Graph 和流水槽的初始化为 `511.862 / 700.688 ms`，allocated 增量为 `168384512 / 489719296` 字节，三路测量复用这些对象。

### Nsight 重叠分析

诊断 Run `di02-20261023-c-profile` 使用 Nsight Systems `2024.1.1.0`，B4/C、512输入/256输出，采集预热后的16个推进窗口（10—25）。以 `pipeline_forward:seq=j` 找到 Graph 发射，通过 correlationId 关联实际 kernel/memset 活动并集，再与 `pipeline_post:seq=j−1` 的 CPU 范围求交，排除 `pipeline_wait`。

| 采集内指标 | 结果 |
|---|---|
| 当前轮实际 Graph 活动与上一轮 CPU post 有交集 | 16/16 对；去掉前结果在窗口前产生的首对后仍为15/15 |
| 交集时长 | 中位46.410 µs，范围36.606—56.920 µs |
| 上一轮 CPU post 时长 | 中位47.520 µs |
| post 完成后当前 Graph 剩余跨度 | 至少18.646 ms |

### 复现入口与输入指纹

精确测量源码为本节标识的完整SHA；当前main保留相同基准入口。

输入为 UTF-8 JSON，顶层 `prompts` 数组至少4条，每条为512个模型词表内的整数 Token ID。程序在计时外读取并转为 CUDA 张量，B1使用第一条，B4使用前四条。文件字节参与 `input_file_sha256`，前四条规范化 ID 参与 `source_ids_sha256`；`templates` 和 `template_sequence_sha256` 标识输入模板及输出上限。

实测文件 SHA256：`3262a3904a3c8dfa6d85b8cd63bdc64647783a0d5cd4604496c5166b6362dd80`；前四条 ID 摘要：`22e0ff4cf8fd7422ecd69d068d732d196bd46fb9fc278e3a4ee1df40faf1291a`。模板序列摘要为 B1 `729771e7b2826495ec0bc8e802a4f94110eac5061e68429a5f6754952bf60e29`、B4 `8e19ee265caaacaf16c15d4532d3573ffe776cb822d76b865b22192c1b910aa0`。

将 `INPUT_JSON` 替换为自己准备的文件；`mktemp` 创建仓库外的新结果目录，避免覆盖已有证据：

```bash
export OMP_NUM_THREADS=1
INPUT_JSON=/absolute/path/inputs.json
BENCH_DIR=$(mktemp -d /tmp/amperekv-di02-XXXXXX)
python -m ampere_kv.check_async_decode
for B in 1 4; do
  python -m ampere_kv.bench_workload --mode steady --batch "$B" --cuda-graph \
    --pipeline abc --output-tokens 256 --repeats 3 --input "$INPUT_JSON" \
    --output "$BENCH_DIR/abc-b$B.json" --run-id "di02-repro-abc-b$B"
done
```

结果 JSON 保存提交、配置、输入/输出摘要、前向与 Graph/丢弃计数、显存和回收情况。先核对工作量，再比较 `total_s`。

### 技术来源

本项目独立实现固定请求身份的单步流水，参考 [DeepInfer 固定提交774ce244](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01) 的 [`worker.execute_model_step`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py)（current-forward / previous-post）及 [`prepare_tensor_input_zos.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py)（设备 Token 接续）。参考项目使用 [Apache-2.0 许可证](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/LICENSE)。

## 同步结构化输出（6ea7014）

实测源码为 `6ea7014d03c8d856c679e74a5230ebab3deff4d4`，Run ID 为 `di05b-paired-01`。同一 RTX 3090，PyTorch `2.5.1+cu124`、CUDA `12.4`、Transformers `4.51.0`、XGrammar `0.1.18`；固定 Qwen/Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`。

输入长度为 B1 `[25]`、B4 `[25,26,26,27]` Token，使用仅含必需布尔字段 `approved`、不允许额外字段的 APPROVAL JSON Schema。三路径共用输入、Schema 和请求参数：温度 `0.8`、K=`40`、seed=`42+i`（i从0起），生成上限64、自然EOS；每请求实际输出6 Token，包含EOS。

`eager-reference` 使用同步eager前向与参考采样；`graph-reference` 使用同步Graph与参考采样；`graph-cuda` 使用同步Graph与CUDA采样。Graph沿用固定B1/B4，其他批量回退eager；CUDA沿用同K批量快路径及异构回退。每路径预热1次，按三路径正序→逆序→正序各测1组，共18个正式样本。计时从submit到整批输出已回宿主且资源释放，包含matcher初始化、逐步掩码/传输/采样/提交；模型加载、输入准备、语法编译、建池与Graph捕获在计时外，JSON校验和文本解码在计时后。

| B | 路径 | 三次整批完成时间（ms） | 中位数（ms） |
|---:|---|---|---:|
| 1 | eager-reference | 189.628, 189.971, 190.602 | 189.971 |
| 1 | graph-reference | 155.336, 155.542, 155.292 | 155.336 |
| 1 | graph-cuda | 154.168, 154.275, 154.398 | 154.275 |
| 4 | eager-reference | 405.230, 410.247, 406.784 | 406.784 |
| 4 | graph-reference | 375.447, 374.695, 376.360 | 375.447 |
| 4 | graph-cuda | 371.166, 371.141, 372.669 | 371.166 |

按未舍入中位数计算，Graph相对eager的耗时下降为B1 **18.23%**、B4 **7.70%**；Graph后增加CUDA采样再下降 **0.68%/1.14%**（1.060/4.280ms），合计相对eager下降 **18.79%/8.76%**。结果对应上述短输入、紧凑JSON与同步执行配置。

三路径在同批量的完整输出和工作量一致，均生成合法 `{"approved":true}`、正常EOS并完成回收。B1的5次Decode全部命中Graph；B4共20个Decode Token、8次前向，其中B1/B4各2次命中、B2/B3各2次回退，仍有四次整段Prefill。

初始化单列：B1/B4语法编译分别83.576/83.226ms；建池与捕获的墙钟按eager-reference、graph-reference、graph-cuda顺序分别为B1 `1.263/503.618/203.608 ms`、B4 `1.617/419.308/417.436 ms`。这些初始化不计入表中正式样本。

按README准备main、固定模型缓存和XGrammar依赖后，运行既有入口；结果路径须在仓库外：

```bash
python -m ampere_kv.check_structured_output --model
python -m ampere_kv.bench_structured_output --batches 1 4 \
  --output /absolute/path/structured.json --run-id structured-repro-01
```

## 动态到达负载与分块取舍

固定RTX3090、Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`、Torch2.5.1+cu124/CUDA12.4/transformers4.51.0，BF16/V1、greedy、每请求512输入/32输出、忽略EOS。四条输入模板沿用上文B4指纹；速率固定为`2.128913764807336`请求/s，steady按间隔到达、burst每8请求成组，各60秒128请求，最多排空120秒。max_batch=4、每层280块、Graph容量544，mixed/流水/结构化关闭。每进程4请求预热、16请求闭环参考在外；显式速率不随参考mu变化。三对按AB/BA/AB运行，每对先算B/A变化，再取变化的中位数；TTFT与完成延迟从计划到达计，Token间隔按commit入口就绪时间计，窗口与含排空吞吐分列。

| 对照/提交 | 负载 | Token间隔中位变化 | 间隔P95变化 | TTFT中位变化 | 完成延迟中位变化 |
|---|---|---:|---:|---:|---:|
| B1/B4→B1至B4 Graph，`e76d721` | steady | −25.16% | −0.28% | −6.40% | −16.77% |
| B1/B4→B1至B4 Graph，`e76d721` | burst | +0.03% | +0.004% | −1.93% | −1.55% |
| 整段→256分块，两侧B1至B4 Graph，`e6325dc` | steady | +2.52% | −24.02% | +42.34% | +15.03% |
| 整段→256分块，两侧B1至B4 Graph，`e6325dc` | burst | +0.21% | −21.23% | +42.74% | +16.59% |

Run ID 为 `load02-pair{1,2,3}-{a,b}01` 和 `load04-pair{1,2,3}-{a,b}01`。各进程分别运行一次 steady 和 burst，A/B 对应表中箭头两端。两组分别核对输入、完整到达表、逐模板输出摘要、工作量和回收。LOAD02 的 steady Graph 调用覆盖从 2.14%—5.71% 升至 100%，捕获增加 17.75 MiB allocated 和 44 MiB reserved，捕获墙钟时间的配对增量中位数为 380.124 ms。LOAD04 的 Graph 命中率为 100%，回退为 0，12 场共完成 1536 个请求。在该固定到达率下，steady 窗口输出吞吐变化中位数为 −0.345%，排空多耗时 81.5—86.8 ms，burst 窗口输出吞吐持平。

256 Token 分块在两种负载中都降低了 Token 间隔 P95，同时延长了首 Token 和整请求完成时间。每场 3968 个间隔中，超过 100 ms 的数量在 steady 中从 253 增至 506，在 burst 中从 448 增至 896。burst 的准入等待中位数也增加了 40.34%。后续块需要再次执行完整层栈并读回历史。根据这两组负载的取舍，默认使用整段 Prefill，256 Token 分块可显式开启。

当前复现入口如下；用自备四条512 Token输入时记录新指纹和Run ID，复现原始配对则使用对应归档输入与提交：

```bash
python -m ampere_kv.bench_arrivals --input /absolute/path/inputs.json \
  --output /absolute/path/arrivals.json --run-id arrivals-repro-01 \
  --cuda-graph --graph-batches 1 2 3 4 --prefill-chunk-size 0 \
  --scenarios steady burst --seconds 60 --rate 2.128913764807336
```

将块长改为256运行另一独立进程；B1/B4对照显式传`--graph-batches 1 4`。阶段定位沿用`profile_arrivals --capture`及`--mode insert`；记录与Profiler采集用于解释机制，正式配对不启用它们。

## 复现说明

按[README](../README.MD)准备固定模型、兼容环境和扩展。算子入口为`paged_decode --benchmark`，离线整批为`bench_external`，流水为`bench_workload`，动态负载为`bench_arrivals`，投机为`bench_speculative_decode`。

结果路径使用仓库外的绝对路径。JSON 保存源码 SHA、配置、原始样本、输入与输出指纹、Graph 与投机计数以及回收结果。使用自备输入时记录新指纹和 Run ID。各节列出的 SHA 对应当时的测量源码，历史原始证据保存在私有归档中。
