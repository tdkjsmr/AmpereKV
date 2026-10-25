"""同步结构化输出三路径配对；编译/建池/捕获在计时外，提交到回收计入。"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from ampere_kv.check_structured_output import APPROVAL, PATHS, validate
from ampere_kv.runner import MODEL_ID, MODEL_REVISION, encode_prompt, load_model_and_tokenizer
from ampere_kv.scheduler import FINISHED, Scheduler
from ampere_kv.structured_output import SchemaCompiler


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def execute(scheduler, prompts, grammar, backend, tokenizer):
    before = (scheduler.graph_calls, scheduler.graph_fallbacks,
              {b: g.replays for b, g in (scheduler.graphs or {}).items()})
    torch.cuda.synchronize()
    started = time.perf_counter()
    requests = [scheduler.submit(f"r{i}", prompt, 64, structured=grammar, temperature=0.8,
                                 top_k=40, seed=42 + i, sampling_backend=backend)
                for i, prompt in enumerate(prompts)]
    decode_batches = {}
    while any(request.status != FINISHED for request in requests):
        _, sizes = scheduler.step()
        for size in sizes:
            decode_batches[size] = decode_batches.get(size, 0) + 1
    elapsed = (time.perf_counter() - started) * 1000
    # 校验与文本解码均在计时后；结束Token已取回，retire已释放物理块与语法状态。
    outputs = [r.output_ids for r in requests]
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    for request, text in zip(requests, texts):
        validate(text, APPROVAL)
        assert request.structured_complete and request.stop_reason == "EOS", (text, request.stop_reason)
        assert request.sampler.structured is request.sampler.uniforms is None
    reclaimed = (not scheduler.waiting and not scheduler.running and scheduler.prefilling is None
                 and scheduler.reserved_blocks == 0
                 and all(s._pool.num_free_blocks == scheduler.blocks_per_layer for s in scheduler.storages))
    assert reclaimed
    graph = dict(calls=scheduler.graph_calls - before[0], fallbacks=scheduler.graph_fallbacks - before[1],
                 captures=scheduler.graph_captures,
                 replays={b: g.replays - before[2][b] for b, g in (scheduler.graphs or {}).items()})
    return dict(ms=elapsed, output_ids=outputs, output_sha256=digest(outputs),
                output_tokens=[len(ids) for ids in outputs], texts=texts, json_valid=True,
                stop_reasons=[r.stop_reason for r in requests],
                decode_tokens=sum(len(ids) - 1 for ids in outputs), decode_batches=decode_batches,
                graph=graph, reclaimed=reclaimed)


def measure(model, tokenizer, compiler, batch):
    texts = ('Approve this request. Return only JSON with boolean field approved.',
             'Is a routine meeting acceptable? Return only JSON with boolean field approved.',
             'Approve reading a book. Return only JSON with boolean field approved.',
             'Decide on a harmless request. Return only JSON with boolean field approved.')
    prompts = [encode_prompt(tokenizer, text) for text in texts[:batch]]
    width = ((max(p.shape[1] for p in prompts) + 64 + 15) // 16) * 16
    capacity = sum((p.shape[1] + 64 + 16) // 16 for p in prompts)
    started = time.perf_counter()
    grammar = compiler.compile(APPROVAL)
    compile_ms = (time.perf_counter() - started) * 1000
    schedulers, initialization = {}, {}
    for path in PATHS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        scheduler = Scheduler(model, capacity, max_batch=batch)
        if path != "eager-reference":
            scheduler.enable_graphs(width)
        torch.cuda.synchronize()
        schedulers[path] = scheduler
        initialization[path] = (time.perf_counter() - started) * 1000
    warmups, samples = {}, {path: [] for path in PATHS}
    for path in PATHS:
        warmups[path] = execute(schedulers[path], prompts, grammar,
                                "cuda" if path == "graph-cuda" else "reference", tokenizer)
    for order in (PATHS, PATHS[::-1], PATHS):
        for path in order:
            sample = execute(schedulers[path], prompts, grammar,
                             "cuda" if path == "graph-cuda" else "reference", tokenizer)
            warmup = warmups[path]
            assert sample["output_ids"] == warmup["output_ids"], (path, "同配置复跑不同")
            assert sample["graph"] == warmup["graph"], (path, "Graph计数变化")
            if path != "eager-reference":
                assert sample["graph"]["calls"] > 0 and sample["graph"]["replays"].get(batch, 0) > 0
            samples[path].append(sample)
    medians = {path: statistics.median(s["ms"] for s in rows) for path, rows in samples.items()}
    workloads = [(rows[0]["output_tokens"], rows[0]["stop_reasons"], rows[0]["decode_tokens"], rows[0]["decode_batches"])
                 for rows in samples.values()]
    paired = all(workload == workloads[0] for workload in workloads)
    ratios = ({"eager-reference_over_graph-reference": medians[PATHS[0]] / medians[PATHS[1]],
               "graph-reference_over_graph-cuda": medians[PATHS[1]] / medians[PATHS[2]],
               "eager-reference_over_graph-cuda": medians[PATHS[0]] / medians[PATHS[2]]} if paired else None)
    for path in PATHS:
        print(f"[配对] B={batch} {path} 样本ms={[round(s['ms'], 3) for s in samples[path]]} "
              f"中位数={medians[path]:.3f} 输出数={samples[path][0]['output_tokens']} "
              f"摘要={samples[path][0]['output_sha256']} Graph={samples[path][0]['graph']}")
    print(f"[工作量] B={batch} 三路径一致={paired}；速度比={ratios if paired else '不计算，输出数或Decode工作量不同'}")
    return dict(batch=batch, input_tokens=[p.shape[1] for p in prompts],
                input_sha256=digest([p[0].tolist() for p in prompts]),
                compile_ms=compile_ms, initialization_ms=initialization, width_tokens=width,
                blocks_per_layer=capacity, warmups=warmups, samples=samples,
                medians_ms=medians, same_workload=paired, speed_ratios=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batches", type=int, nargs="+", choices=(1, 4), default=(1, 4))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("结果JSON请写入仓库外目录")
    model, tokenizer = load_model_and_tokenizer()
    stops = model.generation_config.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    compiler = SchemaCompiler(tokenizer, model.config.vocab_size, stops)
    config = dict(schema=APPROVAL, temperature=0.8, top_k=40, seeds=[42, 43, 44, 45],
                  max_new_tokens=64, ignore_eos=False, paths=PATHS,
                  order=[list(PATHS), list(PATHS[::-1]), list(PATHS)], warmup=1, repeats=3)
    result = dict(run_id=args.run_id,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                   cuda=torch.version.cuda, xgrammar=version("xgrammar"),
                                   transformers=version("transformers")),
                  model=dict(id=MODEL_ID, revision=MODEL_REVISION), config=config,
                  config_sha256=digest(config), results=[])
    with torch.inference_mode():
        for batch in dict.fromkeys(args.batches):
            result["results"].append(measure(model, tokenizer, compiler, batch))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] {args.run_id}，JSON={args.output}；计时含submit到整批回收，编译/建池/捕获另列")


if __name__ == "__main__":
    main()
