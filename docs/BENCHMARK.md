# 固定负载性能与执行取舍

本报告记录RTX3090上的外部整批对比、Graph与流水消融、动态到达延迟取舍。整批完成时间、动态请求延迟与Nsight诊断分别列出配置和统计。

历史实验的提交哈希标识对应测量版本；归档源码与输入的获取要求见文末复现说明。

## 测量定义

- GPU：同一台 RTX 3090（24 GiB，驱动 `595.71.05`）；模型为 `Qwen/Qwen3-8B`，Revision `b968826d9c46dd6066d109eabc6255188de91218`，BF16 权重。AmpereKV 使用 BF16 KV，vLLM 配置 `kv_cache_dtype=auto`。版本分别为 AmpereKV `914c026` 与 vLLM `0.29.0`，两引擎独立环境、分时驻留。
- 输入为同一份 Token ID 文件；B=1/B=4，每请求 512 输入、固定生成 32 Token，greedy、忽略 EOS、关闭前缀复用。逐 Run 核对输入指纹。
- 各进程预热 1 次，测量 5 次；计时从输入提交到全部输出 ID 回到宿主侧。模型加载、分词、输入准备、预热和建池在计时外。AmpereKV 接收预建 CUDA 张量，vLLM 接收宿主 Token ID 列表。
- 请求配置：AmpereKV 为 BF16/V1、块大小 16、整段 Prefill、`mixed=False`、无自建 Graph；vLLM 请求 `max_model_len=1024`、`max_num_seqs=4`、`kv_cache_dtype=auto`，采样器设置 `VLLM_USE_FLASHINFER_SAMPLER=0`。vLLM 的实际后端、Graph/Chunk 和 KV 预算以对应成功进程日志为准。
- 每个 B 按 AmpereKV→vLLM→vLLM→AmpereKV 运行，分别报告五样本、中位数和极差，并计算同顺序配对的完成时间比。

## Eager 路径与 vLLM 对比（914c026）

| B | 顺序 | 引擎 | Run ID | 五次完成时间（ms） | 中位数（ms） | 范围（ms） | 输出吞吐中位数（Token/s） |
|---:|:---:|:---|:---|:---|---:|---:|---:|
| 1 | A→V | AmpereKV | g7-b1-a1 | 1219.776, 1214.810, 1213.335, 1203.485, 1207.982 | 1213.335 | 16.291 | 26.374 |
| 1 | A→V | vLLM | g7-b1-v1 | 745.840, 743.050, 744.809, 744.045, 744.312 | 744.312 | 2.790 | 42.993 |
| 1 | V→A | vLLM | g7-b1-v2 | 741.343, 745.636, 742.076, 739.152, 741.014 | 741.343 | 6.484 | 43.165 |
| 1 | V→A | AmpereKV | g7-b1-a2 | 1216.639, 1213.374, 1248.568, 1221.421, 1222.408 | 1221.421 | 35.194 | 26.199 |
| 4 | A→V | AmpereKV | g7-b4-a1 | 1787.912, 1762.931, 1762.056, 1773.577, 1833.220 | 1773.577 | 71.164 | 72.171 |
| 4 | A→V | vLLM | g7-b4-v1 | 1097.150, 1057.032, 1100.921, 1058.886, 1058.963 | 1058.963 | 43.889 | 120.873 |
| 4 | V→A | vLLM | g7-b4-v2 | 1103.302, 1104.227, 1064.000, 1064.046, 1106.675 | 1103.302 | 42.675 | 116.015 |
| 4 | V→A | AmpereKV | g7-b4-a2 | 1775.248, 1824.464, 1790.758, 1792.871, 1798.857 | 1792.871 | 49.216 | 71.394 |

同顺序配对的 AmpereKV/vLLM 完成时间中位数比：B=1 为 `1.630×`、`1.648×`，B=4 为 `1.675×`、`1.625×`。反向顺序的同引擎中位数变化：B=1 AmpereKV `0.67%`、vLLM `0.40%`；B=4 AmpereKV `1.09%`、vLLM `4.19%`，均低于预设的 `5%` 环境复核线。

八个 Run 均报告每请求 512 输入、32 输出；输入文件 SHA256 为 `3262a3904a3c8dfa6d85b8cd63bdc64647783a0d5cd4604496c5166b6362dd80`，选中 ID 的 SHA256：B=1 为 `e2d1ac38a473406a5e9a26f46a65ba54941fa9eaba11d09d76bf550b1942e36c`，B=4 为 `22e0ff4cf8fd7422ecd69d068d732d196bd46fb9fc278e3a4ee1df40faf1291a`。输出数分别为 `[32]` 与 `[32,32,32,32]`；各日志的 `914c026` 是本项目基准入口提交，vLLM 软件版本单独锁定为 `0.29.0`。

八个结果 JSON 记录 Python `3.12.3`、上述 GPU/驱动、模型 Revision 和五个完整样本。AmpereKV 使用 PyTorch `2.5.1+cu124` / CUDA `12.4`，vLLM 使用 PyTorch `2.13.0+cu130` / CUDA `13.0`。

四个 vLLM 测量进程的启动配置一致：Chunked Prefill 启用，`max_num_batched_tokens=4096`；Attention 为 `FLASH_ATTN` / FlashAttention 2；PIECEWISE/FULL Graph 捕获完成；KV 预算 `5.88 GiB`、容量 `42,800 Token`；`VLLM_USE_FLASHINFER_SAMPLER=0`，`max_model_len=1024`，`max_num_seqs=4`。

## 内部消融：固定 RoPE 频率复用

同一 RTX 3090、B=1/B=4、512 输入/32 固定输出；旧提交 `a19a66a` 与新提交 `914c026`，每进程预热 1 次、测量 5 次。顺序旧→新→新→旧，以下只在各顺序对内比较；对应输出指纹不变。

| B | 顺序对 | 旧版五样本（ms）/中位数 | 新版五样本（ms）/中位数 | 新版完成时间降低 |
|---:|:---:|:---|:---|---:|
| 1 | 旧→新 | 1378.741, 1382.144, 1382.681, 1385.718, 1386.522 / 1382.681 | 1208.197, 1205.226, 1208.321, 1210.355, 1207.444 / 1208.197 | 12.62% |
| 1 | 新→旧 | 1390.707, 1378.239, 1372.658, 1376.401, 1374.858 / 1376.401 | 1209.952, 1210.862, 1208.245, 1210.002, 1196.194 / 1209.952 | 12.09% |
| 4 | 旧→新 | 1973.174, 2004.251, 1942.045, 1955.716, 1946.537 / 1955.716 | 1764.673, 1762.904, 1766.771, 1766.382, 1763.794 / 1764.673 | 9.77% |
| 4 | 新→旧 | 1995.330, 2087.706, 1988.565, 1981.235, 1986.881 / 1988.565 | 1796.480, 1798.880, 1812.104, 1798.426, 1797.410 / 1798.426 | 9.56% |

另一次 B=4 单批 Nsight Systems 诊断中，旧→新 `cudaMemcpyAsync` 与 `cudaStreamSynchronize` 均为 `4066→2698` 次，Kernel launch 均为 `84590` 次。减少的 1368 次与 38 次前向 × 36 层吻合。

## Graph 路径开关消融与 vLLM 对比（7e6bf9b）

被测提交为 `7e6bf9b`，与上一组 Eager 外部对比独立。同一 RTX 3090（驱动 `595.71.05`）按 E→G→V→V→G→E 顺序运行：E 为同提交 AmpereKV Graph 关闭，G 为 Graph 开启，V 为 vLLM `0.29.0`；B=1/B=4 均每请求 512 输入、固定 32 输出，预热 1 次、正式测量 5 次。下表记录原始五样本，单位 ms；Run ID 与结果 JSON 的 `run_id` 字段一致。

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

该组 vLLM 启动配置同前：Chunked Prefill（`max_num_batched_tokens=4096`）、FLASH_ATTN/FlashAttention 2、PIECEWISE/FULL Graph、KV 预算 `5.88 GiB`、容量 `42,800 Token`，`VLLM_USE_FLASHINFER_SAMPLER=0`。AmpereKV 使用 PyTorch `2.5.1+cu124` / CUDA `12.4`，vLLM 使用 PyTorch `2.13.0+cu130` / CUDA `13.0`。

计时包含调度、前向、输出取回与回收，初始化在计时外。B1 开启 Graph 时同时切换到批量 B1 前向入口，因此 E/G 衡量的是整条执行路径的变化。

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

## DI-02：受限单步流水消融（f5d5c33）

本节测量 **512 输入 / 256 输出** 的三路流水消融，实测源码为 `f5d5c3351ae6d15bcc6ae5590f29558e7aaf8f7f`。

### 配置与计时

- 日期：2026-10-23；同一 RTX 3090（24 GiB），Qwen/Qwen3-8B，Revision `b968826d9c46dd6066d109eabc6255188de91218`；PyTorch `2.5.1+cu124`、CUDA `12.4`。Windows 只做源码和证据分析，实际模型及计时在 CUDA 环境执行。
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

B4 流水外有5次回退 Decode、11个 Token，合计1020个 Decode Token，加4个 Prefill 首 Token 为1024。正式样本无重新捕获、丢弃或额外 GPU 前向，C 的额外宿主推进用于启动与排空。同一 B 的 A/B/C 输出摘要一致；各样本结束后额度归零、36层块全部归还，allocated 回到起始值。

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

例如，Graph13 的设备跨度为64.559504—89.377665 ms，CPU post12 为64.562892—64.615959 ms，实际设备活动交集52.491 µs，显示当前轮 GPU 执行与上一轮 CPU 提交确实重叠。总体收益还包括下一轮准备/提交与前一轮 GPU 执行的交叠；约46 µs的 CPU post 交集是其中一个观察量。

### 复现入口与输入指纹

先按 README 准备兼容 CUDA 环境、固定模型缓存和扩展。固定 `v2.0.1` 不含流水入口；精确源码使用上述完整 SHA。已有干净克隆可执行：

```bash
git fetch origin codex/deepinfer-decode-handoff
git switch --detach f5d5c3351ae6d15bcc6ae5590f29558e7aaf8f7f
```

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

另行采集短时间线：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output="$BENCH_DIR/c-profile" \
  python -m ampere_kv.bench_workload --mode steady --batch 4 --cuda-graph \
    --pipeline c --output-tokens 256 --profile-rounds 10 16 --input "$INPUT_JSON" \
    --output "$BENCH_DIR/c-profile.json" --run-id di02-repro-c-profile
nsys export --type sqlite --output "$BENCH_DIR/c-profile.sqlite" "$BENCH_DIR/c-profile.nsys-rep"
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

按README准备当前开发分支、固定模型缓存和XGrammar依赖后，运行既有入口；结果路径须在仓库外：

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

Run ID为`load02-pair{1,2,3}-{a,b}01`和`load04-pair{1,2,3}-{a,b}01`，各进程含steady/burst一次；A/B分别对应表中箭头两端。两组分别核对输入/完整到达表、逐模板输出摘要、工作量和回收。LOAD02的steady Graph调用覆盖从2.14%—5.71%升至100%，捕获增加17.75MiB allocated、44MiB reserved，墙钟配对增量中位380.124ms。LOAD04全部场景Graph100%命中、零回退，12场1536请求全部完成；steady窗口输出吞吐变化中位−0.345%，排空多81.5—86.8ms，burst窗口吞吐不变。固定到达率下的窗口吞吐不等于最大持续处理能力。

时间线定位版本`e1ff14e`，Run ID为`load03-steady-on01`、`load03-steady-profile01`、`load03-insert01`：无工具记录558个间隔中，33个跨新请求Prefill的长间隔约162—165ms；代表例为Prefill 142.038ms后再执行约25.123ms Decode。Prefill可见GPU工作并集134.342ms；其中宿主同步与GPU工作重叠，不能把等待另加到阶段耗时。Graph内部未展开。固定轮次插入AB/BA将旧请求局部间隔增加138.193/138.826ms，与插入Prefill 137.902/138.428ms相符。

保留整段Prefill默认，256分块作为显式实验。分块把长停顿拆短，但每场3968个间隔中，超过100ms的数量steady为253→506、burst为448→896；该界线仅描述频次。burst准入等待中位还增加40.34%。后续块仍需执行完整层栈并读回历史，这与早期小Chunk变慢的负结果相容；读回改进前后的历史观察不与本表混算，也不据此推断更小Chunk会更快。当前两种负载都没有支持全局开启分块的综合收益。

开发分支的当前复现入口如下；用自备四条512 Token输入时记录新指纹和Run ID，复现原始配对则使用对应归档输入与提交：

```bash
python -m ampere_kv.bench_arrivals --input /absolute/path/inputs.json \
  --output /absolute/path/arrivals.json --run-id arrivals-repro-01 \
  --cuda-graph --graph-batches 1 2 3 4 --prefill-chunk-size 0 \
  --scenarios steady burst --seconds 60 --rate 2.128913764807336
```

将块长改为256运行另一独立进程；B1/B4对照显式传`--graph-batches 1 4`。阶段定位沿用`profile_arrivals --capture`及`--mode insert`；记录与Profiler采集用于解释机制，正式配对不启用它们。

## 复现说明

整批测量入口为 `bench_external`，算子基准为 `paged_decode --benchmark`，流水消融入口和命令见上节。元数据合并上传版本完整标识为 `0ac0d950f31720ce47fa99a834bd98e2e33b5096`。

- 各组数据对应各自的源码、配置与输入，配对比较在组内进行。部分历史源码和原始输入保存在独立归档中；使用自备输入时记录新的 Run ID、提交和结果。
- 整批计时、动态到达统计和Nsight采集分开；初始化耗时与显存状态差单独列出。vLLM的实际后端、Graph和Chunk来自启动日志。
- 原始输入与采集文件保留在私有归档。公开采集文件前需清除环境变量等敏感信息。
