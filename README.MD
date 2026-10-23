# AmpereKV Engine

AmpereKV 是面向 Qwen3-8B、单张 RTX 3090（`sm_86`）的轻量文本推理引擎，提供分页 KV 缓存、CUDA Decode、多请求调度和可选 CUDA Graph。Embedding、线性层与 MLP 等使用 PyTorch。

## 能力与边界

- 单请求支持 BF16 Prefill、贪心生成、连续或分页 KV；`runner --mode generate --cache paged` 的 BF16 Decode 使用 SDPA。多请求调度默认使用 BF16/V1 分页 CUDA Decode，支持固定分块 Prefill 和可选混合前向；可在首次提交请求前启用固定 B1/B4 Decode CUDA Graph，其他批量回退 eager。
- 同步调度仍为默认；固定 B1/B4、BF16/V1 greedy 纯 Decode 可显式选择设备 Token 接续与单步异步流水。它是实验驱动，不是普通生成入口的参数；不支持任意批次重映射。
- 分页缓存按请求维护独立块表，共享每层物理块池；已分配的块归请求独占，完成、合作式取消或到期后归还。块大小固定为 16 Token。
- INT8 仅量化 KV，不量化模型权重：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale。同容量 BF16/INT8 KV 布局字节比约为 `1.9248`，不是实测最大上下文倍率；INT8 多请求、长期质量和最大容量未验证。
- 不提供跨轮会话历史/KV 保留、前缀复用、通用模型适配、多模态或 HTTP 服务。Graph 不支持 INT8/V3、mixed 或动态批次。性能结果仅适用于报告中的设备、提交和负载。

## 工具链

已验证环境：Ubuntu 22.04、RTX 3090（24 GiB）、驱动 `595.71.05`、CUDA Toolkit `12.4`、Python `3.12.3`。Python 依赖为 PyTorch `2.5.1+cu124`、`transformers 4.51.0`、`accelerate 1.6.0`、`huggingface_hub 0.36.2`。构建、生成及模型/调度/Graph 回归已在独立 Python 环境通过，复用已有驱动、Toolkit 和固定模型缓存；未验证全新系统安装或首次模型下载。

需要 Git、`make`、可用的 C++/CUDA 编译器和足够空间存放约 16 GB 的模型权重。`setup.py` 只构建本项目扩展，**不自动安装所有运行依赖**。以下命令供 CUDA 12.4 云端环境依次执行；若使用其他环境，应先确认与 PyTorch 扩展构建兼容：

```bash
git clone https://github.com/tdkjsmr/AmpereKV.git
cd AmpereKV
git switch --detach v2.0.1  # 固定运行版本
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=69' wheel
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'transformers==4.51.0' 'accelerate==1.6.0' 'huggingface_hub[cli]==0.36.2'
export OMP_NUM_THREADS=1
export HF_HOME=/path/to/persistent/huggingface  # 换成有足够空间的持久化目录；运行时保持一致
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
make build
make smoke
python -m ampere_kv.runner --mode generate --cache paged
```

下载模型不需要 GPU；已有固定 Revision 缓存时跳过下载。生成命令会在终端询问一段文本。模型和分词器只从 `HF_HOME` 对应的本地缓存加载。更多模式与参数见 `python -m ampere_kv.runner --help`。

[架构说明](docs/ARCHITECTURE.md) · [固定负载测量报告](docs/BENCHMARK.md)

## 阅读与测试入口

阅读运行主线：`runner.py`（模型前向）→ `scheduler.py`（调度）→ `paged_cache.py`（缓存）→ `graph_decode.py`（可选 Graph）。模型诊断在 `check_model.py`，通过 `runner --mode cuda-check/int8-check --cache paged` 选择其中一种模式。

在 CUDA 环境中运行 `python -m ampere_kv.<模块名>`：`check_scheduler`、`check_chunked`、`check_mixed`、`check_graph_scheduler` 分别检查调度、分块、混合前向与 Graph。离线整批基准使用 `bench_external`，算子调用基准使用 `paged_decode --benchmark`；参数见各自 `--help`。

异步实验代码在 `async_decode.py`，检查入口为 `python -m ampere_kv.check_async_decode`。`bench_workload --mode steady --cuda-graph --pipeline a|b|c|abc` 分别选择原同步、设备接续同步、单步重叠或交错对照；完整命令、输入要求及已测边界见[DI-02 报告](docs/BENCHMARK.md#di-02受限单步流水消融f5d5c33)。此入口需包含该实现的源码，不在上面的固定 `v2.0.1` 标签中。

开发分支可显式随机生成，默认 `temperature=0` 仍为 greedy：

```bash
python -m ampere_kv.runner --mode generate --cache paged --temperature 0.8 --top-k 40 --seed 42
```

随机采样仅覆盖 BF16/V1、非 mixed；调度器用 `submit(..., temperature=0.8, top_k=40, seed=42)`，兼容同步与可选 B/C。`top_k=-1` 表示全词表，或取 `1..vocab_size`；seed为 `0..2^32-1`，省略则随机选定（runner打印，调度器可读 `request.sampler.seed`）。相同 logits/参数/seed/逻辑位置可复现，不保证不同批量的模型 logits 或随机生成序列一致，也不保证跨软硬件版本一致；此功能不在固定 `v2.0.1` 标签中。纯采样检查为 `python -m ampere_kv.check_sampler`，加 `--model` 另测短模型集成。
