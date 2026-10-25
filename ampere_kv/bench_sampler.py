"""采样调用配对：微基准主矩阵，或--model测同一流水模式的模型整批完成。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch

from ampere_kv.sampler import RequestSampler, sample_batch
from ampere_kv.check_sampler import capture_sampling, compare_traces, check_sampling


def paired(functions, warmup, groups, iterations):
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    samples = {name: [] for name in functions}
    for group in range(groups):
        order = list(functions)[::(-1 if group % 2 else 1)]
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / iterations)
    return samples


def micro(args, vocab):
    generator = torch.Generator(device="cuda").manual_seed(91)
    results = []
    for batch in (1, 4):
        for k in (40, 128):
            for dtype in (torch.bfloat16, torch.float32):
                logits = torch.randn(batch, vocab, device="cuda", generator=generator).to(dtype)
                functions = {}
                for backend in ("reference", "cuda"):
                    requests = [SimpleNamespace(output_ids=[], sampler=RequestSampler(
                        1, vocab, "cuda", temperature=0.8, top_k=k, seed=42 + i, sampling_backend=backend))
                        for i in range(batch)]
                    functions[backend] = lambda requests=requests: sample_batch(logits, requests)
                temps = torch.full((batch,), 0.8, device="cuda")
                uniforms = torch.stack([r.sampler.uniforms[0] for r in requests])
                ref, cuda, acceptance = check_sampling(logits, temps, uniforms, k)
                assert torch.equal(functions["reference"](), ref) and torch.equal(functions["cuda"](), cuda)
                samples = paired(functions, args.warmup, args.groups, args.iterations)
                medians = {name: statistics.median(values) for name, values in samples.items()}
                row = dict(batch=batch, vocab=vocab, top_k=k, dtype=str(dtype), acceptance=acceptance,
                           samples_us=samples, medians_us=medians)
                results.append(row)
                print(f"[配对] B={batch} V={vocab} K={k} dtype={dtype} 样本us={samples} 中位us={medians} 边界差异={acceptance['boundary']}")
    return results


def model_bench(args):
    from ampere_kv.bench_external import load_inputs
    from ampere_kv.bench_workload import scheduler_type, run
    from ampere_kv.runner import load_model_and_tokenizer
    prompts, file_hash, source_hash = load_inputs(args.input, 4, 512, False)
    model, _ = load_model_and_tokenizer()
    results = []
    for batch in (1, 4):
        class SampledScheduler(scheduler_type()):
            backend = "reference"

            def submit(self, request_id, *values, **kwargs):
                return super().submit(request_id, *values, **kwargs, temperature=0.8, top_k=40,
                                      seed=42 + int(request_id[1:]), sampling_backend=self.backend)

        start = time.perf_counter()
        scheduler = SampledScheduler(model, batch * 41, max_batch=batch)
        scheduler.enable_graphs(640)
        if args.pipeline != "a":
            scheduler.enable_pipeline(overlap=args.pipeline == "c")
        torch.cuda.synchronize()
        initialization_ms = (time.perf_counter() - start) * 1000
        templates = [dict(id=f"t{i}", input_tokens=512, outputs=128,
                          gpu=torch.tensor([ids], device="cuda", dtype=torch.long)) for i, ids in enumerate(prompts[:batch])]

        def execute(backend, outputs, diagnostic=False):
            scheduler.backend, scheduler.workload_serial = backend, 0
            with capture_sampling() if diagnostic else nullcontext(None) as trace:
                sample = run(scheduler, [t | {"outputs": outputs} for t in templates], batch,
                             requests=batch, pipeline_mode=args.pipeline)
            return (sample, trace) if diagnostic else sample

        def cross_check(samples, outputs):
            if samples[0]["output_sha256"] == samples[1]["output_sha256"]:
                return []
            diagnostics = [execute(backend, outputs, True) for backend in ("reference", "cuda")]
            assert all(s["output_sha256"] == d[0]["output_sha256"] for s, d in zip(samples, diagnostics)), "同后端复跑不同"
            return compare_traces(diagnostics[0][1], diagnostics[1][1])

        short = [execute(backend, 8) for backend in ("reference", "cuda")]
        acceptance = dict(short=cross_check(short, 8))
        warm = [execute(backend, 128) for backend in ("reference", "cuda")]
        acceptance["warm"] = cross_check(warm, 128)
        expected = {name: sample["output_sha256"] for name, sample in zip(("reference", "cuda"), warm)}
        samples = {name: [] for name in ("reference", "cuda")}
        for repeat in range(3):
            for backend in (list(samples) if repeat % 2 == 0 else list(samples)[::-1]):
                sample = execute(backend, 128)
                assert sample["output_sha256"] == expected[backend], "同后端正式样本输出不同，停止计时"
                samples[backend].append(sample)
        times = {name: [s["total_s"] * 1000 for s in values] for name, values in samples.items()}
        medians = {name: statistics.median(values) for name, values in times.items()}
        results.append(dict(batch=batch, pipeline=args.pipeline, initialization_ms=initialization_ms, acceptance=acceptance,
                            input_file_sha256=file_hash, source_ids_sha256=source_hash, samples=samples, medians_ms=medians))
        print(f"[模型] B={batch} 输入512/输出128 temperature=0.8 K=40 seed=42+i "
              f"pipeline={args.pipeline} 初始化ms={initialization_ms:.3f} 样本ms={times} 中位ms={medians}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pipeline", choices=("a", "b", "c"), default="c")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if min(args.warmup, args.groups, args.iterations) < 1 or (args.model and args.input is None):
        parser.error("预热/组数/组内次数须为正；--model需--input")
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(repo)):
        parser.error("结果须写到仓库外的新文件")
    from transformers import AutoConfig
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION
    vocab = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).vocab_size
    metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    model=MODEL_ID, revision=MODEL_REVISION, vocab=vocab,
                    config=vars(args) | {"input": str(args.input), "output": str(args.output)})
    metadata["protocol"] = (dict(input_tokens=512, output_tokens=128, temperature=0.8, top_k=40,
                                 warmup=1, groups=3, short_check_tokens=8) if args.model else
                            dict(temperature=0.8, warmup=args.warmup, groups=args.groups, iterations=args.iterations))
    print(f"[配置] {metadata}")
    print("[口径] 微基准含完整采样调用及输出/临时分配，输入与uniform在计时外，同数据重复读；组末等待。"
          "模型含提交、取回与排空，初始化/预热单列。")
    with torch.inference_mode():
        result = metadata | {"results": model_bench(args) if args.model else micro(args, vocab)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[完成] B1/B4采样配对测量")


if __name__ == "__main__":
    main()
