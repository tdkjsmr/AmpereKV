# AmpereKV Engine

AmpereKV 是面向 Qwen3-8B、单张 RTX 3090（`sm_86`）的轻量文本推理引擎，提供分页 KV 缓存、CUDA Decode、多请求调度、CUDA Graph 与独立草稿投机解码。Embedding、线性层与 MLP 等使用 PyTorch。

## 功能

- **推理与调度**：BF16 Prefill、贪心生成、连续/分页 KV、连续批处理、分块 Prefill、混合前向，以及请求取消和到期回收。
- **CUDA Decode**：直接读取分页 KV，支持等头数和 GQA；调度器默认使用 BF16/V1，单请求 BF16 生成默认使用 SDPA。
- **CUDA Graph** 批量重放 BF16/V1 Decode，批量或序列长度超出捕获范围时回退 eager。
- **INT8 KV**：K 每 32 维一组 scale，V 每 Token、每 KV 头一个 scale；计入 scale 后，同容量 KV 存储比 BF16 减少约 48%。
- **异步流水**：设备端 Token 接续、双缓冲和单步前瞻，提供同步与重叠两种执行方式。
- **投机解码**：Qwen3-0.6B 草稿与 Qwen3-8B 目标，K2/K4、BF16 greedy；草稿支持 reference/CUDA/Graph，Graph 使用设备 Token 连续起草。
- **请求级采样**：temperature、top-k 和独立 seed，兼容同步与异步 Decode。

分页块大小为 16 Token。每个请求持有独立块表，并独占分配给它的物理块。

| 入口 | 支持范围 |
|---|---|
| 单请求生成 | BF16连续/分页KV；INT8使用分页CUDA Decode，权重仍为BF16 |
| 同步调度 | BF16/V1，整段或分块 Prefill。启用 Graph 时默认捕获 B1，批量上限至少为 4 时默认再捕获 B4。可在批量上限内显式选择 B1 至 B4 |
| 混合前向 | BF16、greedy；与Graph分开启用 |
| 异步流水 | BF16/V1、B1/B4 Graph；普通请求，可用随机采样 |
| 投机解码 | 单请求BF16 greedy，K2/K4，独立于多请求调度器 |
| JSON Schema | BF16/V1、整段Prefill、同步eager/Graph、正常EOS；reference/CUDA采样 |

在提交首个请求前捕获 Graph。将捕获容量按 16 Token 对齐，使它覆盖预期的 Decode 历史。Graph 执行模型前向，选词与语法过滤在图外执行。

## 安装与运行

已验证环境：Ubuntu 22.04、RTX 3090（24 GiB）、驱动 `595.71.05`、CUDA Toolkit `12.4`、Python `3.12.3`。Python 依赖为 PyTorch `2.5.1+cu124`、`transformers 4.51.0`、`accelerate 1.6.0`、`huggingface_hub 0.36.2`。

准备 Git、make、C++/CUDA 编译器和至少 20 GB 的模型存储空间（含可选草稿模型），依次执行：

```bash
git clone https://github.com/tdkjsmr/AmpereKV.git
cd AmpereKV
git switch main
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

生成命令读取一段文本，并从 `HF_HOME` 加载固定版本的模型和分词器。已有该版本缓存时可跳过下载。使用 `python -m ampere_kv.runner --help` 查看其他模式。

### 异步流水与随机采样

以下命令以温度 0.8、40 个候选和固定种子生成文本。

```bash
python -m ampere_kv.runner --mode generate --cache paged --temperature 0.8 --top-k 40 --seed 42
```

默认 `temperature=0` 使用贪心生成。`top_k` 取 `1..vocab_size`，或用 `-1` 选择全词表。随机采样的种子范围为 `0..2^32-1`。省略种子时，runner 生成并打印种子。调度器可通过 `submit(..., temperature=0.8, top_k=40, seed=42)` 设置同一组参数。复现采样序列时固定 logits、参数、种子、输出位置、环境和采样后端。

默认采样后端为 `reference`。在生成命令添加 `--sampling-backend cuda`，或在 `submit()` 传入 `sampling_backend="cuda"`，选择分段 Top-K 与小候选融合采样。快路径支持 B1/B4、K40/128、FP32/BF16、V≤524288；异构 K 或混合 greedy 走参考。用 `python -m ampere_kv.check_sampler --sampling-backend cuda --model` 检查，`python -m ampere_kv.bench_sampler` 运行配对微基准。

异步基准用 `bench_workload --mode steady --cuda-graph --pipeline a|b|c|abc` 选择同步、设备接续同步、跨轮重叠或三路配对。完整命令见[流水报告](docs/BENCHMARK.md#单步流水)。

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

同一份语法编译结果可供多个请求使用，每个请求独占 matcher 和掩码。不同 Schema 的请求可以与普通请求同批运行。从首个 Token 开始过滤候选，EOS 表示语法完整结束，达到输出上限时报告“结构化截断”。XGrammar 仅用于结构化请求。

配对入口为 `python -m ampere_kv.bench_structured_output --output /absolute/path/structured.json --run-id structured-01`；配置、原始样本与结果见[同步结构化输出报告](docs/BENCHMARK.md#同步结构化输出6ea7014)。

### 动态到达负载

准备包含四条512 Token输入的JSON后，运行固定到达率负载：

```bash
python -m ampere_kv.bench_arrivals --input /absolute/path/inputs.json \
  --output /absolute/path/arrivals.json --run-id arrivals-01 \
  --cuda-graph --graph-batches 1 2 3 4 --prefill-chunk-size 0 \
  --scenarios steady burst --rate 2.128913764807336
```

基准使用这四条模板，每请求固定生成 32 个 Token。默认用 4 个请求预热，再用 16 个请求做闭环校准。显式传入 `--rate` 可固定正式到达率。默认测量窗口为 60 秒，排空上限为 120 秒。使用 `--scenarios`、`--seconds`、`--repeats` 调整场景、时长和重复次数。每场最多 256 个请求，KV 池按 8 个请求预留，`--max-batch` 控制每次 Decode 最多合批多少请求。

`--prefill-chunk-size 256`选择分块，默认0为整段。Graph省略集合时仍捕获B1/B4，显式集合对应`enable_graphs(capacity, batches=(1, 2, 3, 4))`。组合检查运行`python -m ampere_kv.check_chunked`。延迟、吞吐、排空口径和配对取舍见[动态到达报告](docs/BENCHMARK.md#动态到达负载与分块取舍)。

时间线入口为`python -m ampere_kv.profile_arrivals --input /absolute/path/inputs.json --output /absolute/path/profile.json --run-id profile-01`：默认8秒steady；`--no-record`关闭标记，`--capture`采集预热后的目标批次，`--mode insert`运行固定轮次插入对照。

## 投机解码

先缓存[固定草稿版本](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)，再显式选择草稿后端生成：

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
python -m ampere_kv.speculative_decode --draft-backend graph --draft-tokens 4 --prompt hi
```

两模型共用目标对话模板。草稿默认使用 `reference`，可通过 `cuda` 测量逐步 CUDA Decode，或通过 `graph` 使用设备端连续起草。K 取 2 或 4，最多生成 64 个 Token，默认遇 EOS 停止。目标模型一次验证多个位置，并按实际提交长度更新两套 KV。多 Token 验证与逐 Token 前向在 BF16 近同分处可能选出不同的 Token。

运行以下命令检查与 Hugging Face 逐步生成的一致性，以及投机解码的跨块续写、截断和缓存回收。

```bash
python -m ampere_kv.runner --mode check --cache paged
python -m ampere_kv.check_speculative_decode --model --draft-backend graph --prompt hi --max-new-tokens 64
```

使用 `bench_speculative_decode --compare-graph` 比较六条执行路径。`--mode smoke` 运行冒烟检查，`--mode paired` 对固定 cpp 和 python 输入分别测量。每条路径测三次，每次生成 64 个 Token。完整命令、计时范围和原始样本见[投机解码报告](docs/BENCHMARK.md#投机解码)。

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
