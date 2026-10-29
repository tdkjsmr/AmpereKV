# AmpereKV Engine

AmpereKV 是面向 Qwen3-8B、单张 RTX 3090（`sm_86`）的轻量文本推理引擎，提供分页 KV 缓存、CUDA Decode、多请求调度和可选 CUDA Graph。Embedding、线性层与 MLP 等使用 PyTorch。

## 功能

- **推理与调度**：BF16 Prefill、贪心生成、连续/分页 KV、连续批处理、分块 Prefill、混合前向，以及请求取消和到期回收。
- **CUDA Decode**：直接读取分页 KV，支持等头数和 GQA；调度器默认使用 BF16/V1，单请求 BF16 生成默认使用 SDPA。
- **CUDA Graph**：BF16/V1 Decode默认捕获B1/B4；开发分支可显式增加B2/B3，其他批量或超容量回退eager。
- **INT8 KV**：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale；计入 scale 后，同容量 KV 存储比 BF16 减少约 48%。
- **异步流水**：设备端 Token 接续、双缓冲和单步前瞻，提供同步与重叠两种执行方式。
- **请求级采样**：temperature、top-k 和独立 seed，兼容同步与异步 Decode。

分页块大小为16 Token，每请求独占已分配块。支持组合集中如下：

| 入口 | 支持范围 |
|---|---|
| 单请求生成 | BF16连续/分页KV；INT8使用分页CUDA Decode，权重仍为BF16 |
| 同步调度 | BF16/V1，整段或分块Prefill；Graph默认B1/B4，开发分支可显式B1至B4 |
| 混合前向 | BF16、greedy；与Graph分开启用 |
| 异步流水 | BF16/V1、B1/B4 Graph；普通请求，可用随机采样 |
| JSON Schema | BF16/V1、整段Prefill、同步eager/Graph、正常EOS；reference/CUDA采样 |

Graph容量须16对齐并覆盖历史；首次提交前捕获。模型前向在图内，选词与语法过滤在图外。

## 安装与运行

已验证环境：Ubuntu 22.04、RTX 3090（24 GiB）、驱动 `595.71.05`、CUDA Toolkit `12.4`、Python `3.12.3`。Python 依赖为 PyTorch `2.5.1+cu124`、`transformers 4.51.0`、`accelerate 1.6.0`、`huggingface_hub 0.36.2`。

准备 Git、make、C++/CUDA 编译器和约 16 GB 的模型存储空间，依次执行：

```bash
git clone https://github.com/tdkjsmr/AmpereKV.git
cd AmpereKV
git switch --detach v2.1.0  # 固定发布版本
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=69' wheel
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'transformers==4.51.0' 'accelerate==1.6.0' 'huggingface_hub[cli]==0.36.2'
export OMP_NUM_THREADS=1
export HF_HOME=/path/to/persistent/huggingface  # 模型缓存的持久化目录
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
make build
make smoke
python -m ampere_kv.runner --mode generate --cache paged
```

已有固定 Revision 缓存时可跳过下载。生成命令会询问一段文本；模型和分词器从 `HF_HOME` 的本地缓存加载。其他模式见 `python -m ampere_kv.runner --help`。

### 异步流水与随机采样

完成环境准备后，可运行请求级随机采样：

```bash
python -m ampere_kv.runner --mode generate --cache paged --temperature 0.8 --top-k 40 --seed 42
```

默认 `temperature=0` 为贪心生成。`top_k=-1` 使用全词表，其余取值为 `1..vocab_size`；seed 为 `0..2^32-1`，省略时自动生成并由 runner 打印。调度器使用 `submit(..., temperature=0.8, top_k=40, seed=42)`。固定环境和采样后端下，相同 logits、参数、seed 和输出位置可复现；不同后端可能在累计概率（CDF）分界附近选中不同候选。

默认采样后端为 `reference`。在生成命令添加 `--sampling-backend cuda`，或在 `submit()` 传入 `sampling_backend="cuda"`，选择分段 Top-K 与小候选融合采样。快路径支持 B1/B4、K40/128、FP32/BF16、V≤524288；异构 K 或混合 greedy 走参考。用 `python -m ampere_kv.check_sampler --sampling-backend cuda --model` 检查，`python -m ampere_kv.bench_sampler` 运行配对微基准。

异步基准用 `bench_workload --mode steady --cuda-graph --pipeline a|b|c|abc` 选择同步、设备接续同步、跨轮重叠或三路配对。完整命令见[单步流水报告](docs/BENCHMARK.md#di-02受限单步流水消融f5d5c33)。

### JSON Schema 约束输出

可选依赖为 `xgrammar==0.1.18`（extra 名 `structured`）。在已有环境中安装该固定版本后，用 `python -m ampere_kv.check_structured_output` 检查 tokenizer、掩码和CUDA采样，添加 `--model` 检查三条同步模型路径；`--path eager-reference|graph-reference|graph-cuda` 可指定一条。

```python
import torch

from ampere_kv.runner import load_model_and_tokenizer, encode_prompt
from ampere_kv.scheduler import Scheduler
from ampere_kv.structured_output import SchemaCompiler

model, tokenizer = load_model_and_tokenizer()
scheduler = Scheduler(model, blocks_per_layer=32, max_batch=1)
compiler = SchemaCompiler(tokenizer, model.config.vocab_size, scheduler.eos_ids)
grammar = compiler.compile({"type": "object", "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"], "additionalProperties": False})
with torch.inference_mode():
    scheduler.enable_graphs(256)
    request = scheduler.submit("json", encode_prompt(tokenizer, "只返回JSON，字段approved为布尔值。"),
                               64, structured=grammar, temperature=0.8, top_k=40, seed=42,
                               sampling_backend="cuda")
    while request.status != "finished":
        scheduler.step()
print(tokenizer.decode(request.output_ids, skip_special_tokens=True), request.stop_reason)
```

编译结果可复用，每请求独占matcher与掩码，允许不同Schema和普通请求共存。从首Token起过滤，EOS表示语法完整结束；达到输出上限时报告“结构化截断”。普通请求无需安装XGrammar。

配对入口为 `python -m ampere_kv.bench_structured_output --output /absolute/path/structured.json --run-id structured-01`；配置、原始样本与结果见[同步结构化输出报告](docs/BENCHMARK.md#同步结构化输出6ea7014)。

### 动态到达负载

以下入口位于开发分支`codex/dynamic-load`，测量版本为`e6325dc`；安装步骤中的固定发布版`v2.1.0`不含这些新增入口。已有环境可用以下命令切到测量版本：

```bash
git fetch origin codex/dynamic-load
git switch --detach e6325dc3b934c57afeb25b281b2866ba31391055
python -m ampere_kv.bench_arrivals --input /absolute/path/inputs.json \
  --output /absolute/path/arrivals.json --run-id arrivals-01 \
  --cuda-graph --graph-batches 1 2 3 4 --prefill-chunk-size 0 \
  --scenarios steady burst --rate 2.128913764807336
```

输入为四条512 Token模板，每请求固定32输出。默认4请求预热、16请求闭环校准，显式`--rate`固定正式到达率；窗口60秒、排空上限120秒。`--scenarios`、`--seconds`、`--repeats`选择场景、时长和重复；每场最多256请求，KV池预算8请求，`--max-batch`限制一次Decode组大小。

`--prefill-chunk-size 256`选择分块，默认0为整段。Graph省略集合时仍捕获B1/B4，显式集合对应`enable_graphs(capacity, batches=(1, 2, 3, 4))`。组合检查运行`python -m ampere_kv.check_chunked`。延迟、吞吐、排空口径和配对取舍见[动态到达报告](docs/BENCHMARK.md#动态到达负载与分块取舍)。

时间线入口为`python -m ampere_kv.profile_arrivals --input /absolute/path/inputs.json --output /absolute/path/profile.json --run-id profile-01`：默认8秒steady；`--no-record`关闭标记，`--capture`采集预热后的目标批次，`--mode insert`运行固定轮次插入对照。

## 阅读与测试

执行主线：`runner.py`（模型前向）→ `scheduler.py`（调度）→ `paged_cache.py`（缓存）→ `graph_decode.py`（Graph）。`async_decode.py` 管理跨轮流水，`sampler.py` 管理请求级采样。

通过 `python -m ampere_kv.<模块名>` 运行检查：

| 模块 | 内容 |
|---|---|
| `check_scheduler` / `check_chunked` / `check_mixed` | 调度、分块与混合前向 |
| `check_graph_scheduler` / `check_async_decode` | Graph 与跨轮流水 |
| `check_sampler` | 采样；添加 `--model` 运行模型集成 |
| `runner --mode cuda-check/int8-check --cache paged` | 选择其中一种模式运行模型对照 |

整批基准使用 `bench_external`，算子基准使用 `paged_decode --benchmark`。

[架构说明](docs/ARCHITECTURE.md) · [性能报告](docs/BENCHMARK.md)
