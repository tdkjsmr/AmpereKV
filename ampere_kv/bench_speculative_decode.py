"""单请求五路径配对；正式墙钟与独立阶段诊断分开运行。"""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from ampere_kv.runner import MODEL_ID, MODEL_REVISION, encode_prompt, model_forward
from ampere_kv.scheduler import FINISHED, Scheduler
from ampere_kv.speculative_decode import (DRAFT_ID, DRAFT_REVISION, forward_tokens,
                                          generate_speculative, load_pair, make_caches)

PROMPTS = (
    ("cpp", "请解释 C++ 中 std::vector 和 std::list 的内存布局、随机访问与插入删除复杂度，并说明各自适用场景。"),
    ("python", "请编写一个 Python 函数，使用双指针合并两个已排序的整数列表，并给出一个调用示例。"),
    ("principle", "请用中文解释 Transformer 自注意力中 Q、K、V 的作用，以及因果掩码为什么能防止读取未来信息。"),
)
PATHS = ("reference", "spec-k2", "spec-k4", "graph-sync", "graph-overlap")
OUTPUT_TOKENS = 64


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def top2(logits):
    scores, ids = logits.float().topk(2)
    return {"ids": ids.tolist(), "scores": scores.tolist()}


class Stages:
    """仅诊断使用；Events是流上区间，可包含宿主供给间隙，不是内核忙碌时间之和。"""

    def __init__(self):
        self.records = []

    def start(self, name, gpu=False):
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if gpu else None
        if events:
            events[0].record()
        self.active = name, time.perf_counter(), events

    def end(self):
        name, started, events = self.active
        if events:
            events[1].record()
        self.records.append((name, (time.perf_counter() - started) * 1000, events))

    def report(self):
        # 调用方已在整请求末尾统一等待；这里不逐阶段同步。
        totals = {}
        for name, host, events in self.records:
            row = totals.setdefault(name, {"calls": 0, "host_ms": 0.0, "event_ms": None})
            row["calls"] += 1
            row["host_ms"] += host
            if events:
                row["event_ms"] = (row["event_ms"] or 0.0) + events[0].elapsed_time(events[1])
        return totals


@torch.inference_mode()
def reference(model, ids, *, inspect_at=None):
    """与投机同类的每请求分页池；正式路径无top2、observer或逐步检查。"""
    caches = make_caches(model, ids.shape[1] + OUTPUT_TOKENS)
    outputs, current, candidate = [], ids, None
    try:
        for position in range(OUTPUT_TOKENS):
            logits = forward_tokens(model, current, caches)
            outputs.append(logits[0, -1].argmax().item())
            if position == inspect_at:
                candidate = top2(logits[0, -1])
                break
            current = ids.new_tensor([[outputs[-1]]])
        lengths = [c.length for c in caches]
    finally:
        for cache in caches:
            cache.release()
    return dict(output_ids=outputs, stop_reason="达到生成上限", final_kv_lengths={"target": lengths},
                recovered=all(c.length == 0 and c._pool.num_free_blocks == c._key.shape[0] for c in caches),
                candidate=candidate)


class BenchScheduler(Scheduler):
    def _retire(self, request):
        self.final_lengths = [cache.length for cache in request.caches]
        super()._retire(request)


def initialize_graph(model, inputs):
    maximum = max(item["gpu"].shape[1] for item in inputs) + OUTPUT_TOKENS
    capacity = (maximum + 15) // 16 * 16
    blocks = (maximum + 16) // 16
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    scheduler = BenchScheduler(model, blocks, max_batch=1)
    pool_bytes = torch.cuda.memory_allocated() - before
    scheduler.enable_graphs(capacity, batches=(1,))
    scheduler.enable_pipeline(overlap=True)
    torch.cuda.synchronize()
    return scheduler, dict(wall_ms=(time.perf_counter() - started) * 1000,
                           allocated_delta_bytes=torch.cuda.memory_allocated() - before,
                           pool_allocated_bytes=pool_bytes, capacity=capacity, blocks_per_layer=blocks,
                           captured_batches=sorted(scheduler.graphs), captures=scheduler.graph_captures)


def graph_counts(scheduler):
    return {"calls": scheduler.graph_calls, "fallbacks": scheduler.graph_fallbacks,
            "replays": scheduler.graphs[1].replays, "captures": scheduler.graph_captures}


def graph_request(scheduler, ids, overlap):
    pipeline = scheduler.decode_pipeline
    pipeline.stats["max_inflight"] = 0
    before, stats = graph_counts(scheduler), pipeline.stats.copy()
    request = scheduler.submit("paired", ids, OUTPUT_TOKENS, ignore_eos=True)
    try:
        while request.status != FINISHED:
            pipeline.advance() if overlap else scheduler.step()
    finally:
        if request.status != FINISHED:
            scheduler.cancel(request)
            pipeline.drain()
    after = graph_counts(scheduler)
    return dict(output_ids=request.output_ids, stop_reason=request.stop_reason,
                final_kv_lengths={"target": scheduler.final_lengths},
                recovered=not scheduler.reserved_blocks and not scheduler.waiting and not scheduler.running
                and scheduler.prefilling is None and pipeline.pending is None
                and all(s._pool.num_free_blocks == scheduler.blocks_per_layer for s in scheduler.storages),
                graph={key: after[key] - before[key] for key in ("calls", "fallbacks", "replays")},
                captures=after["captures"], capture_delta=after["captures"] - before["captures"],
                pipeline={key: value - stats.get(key, 0) for key, value in pipeline.stats.items()
                          if key != "max_inflight"}, max_inflight=pipeline.stats["max_inflight"])


def speculative_counts(result, k):
    rounds = result["rounds"]
    rows = []
    for event in rounds:
        count, kept, matched = len(event["proposals"]), event["accepted"], event["matched"]
        rows.append(dict(proposed=count, matched=matched, accepted=kept,
                         correction=int(matched < count), bonus=int(matched == count and count > 0),
                         ordinary=int(count == 0), repair=int(matched == count),
                         committed=len(event["committed"]), short=count < k))
    totals = Counter()
    for row in rows:
        for key, value in row.items():
            totals[key] += value
    if (totals["committed"] + 1 != OUTPUT_TOKENS or totals["proposed"] != result["proposed"]
            or totals["accepted"] != result["accepted"] or len(rows) != result["verification_rounds"]
            or 1 + len(rows) != result["target_forwards"]
            or totals["accepted"] + totals["correction"] + totals["bonus"] + totals["ordinary"] != 63
            or 1 + totals["proposed"] + totals["repair"] != result["draft_forwards"]):
        raise AssertionError("投机前向、补算、接受与提交计数不闭合")
    return dict(totals=totals, rounds=rows,
                short_rounds=[dict(index=i, remaining=OUTPUT_TOKENS - event["output_start"], **row)
                              for i, (event, row) in enumerate(zip(rounds, rows)) if row["short"]],
                target_forwards=result["target_forwards"],
                draft_forwards=result["draft_forwards"], verification_rounds=len(rows))


def check_result(path, result, length):
    if len(result["output_ids"]) != OUTPUT_TOKENS or result["stop_reason"] != "达到生成上限":
        raise AssertionError("固定64输出的工作量或停止原因不符")
    if not result["recovered"] or any(value != length + OUTPUT_TOKENS - 1
                                      for lengths in result["final_kv_lengths"].values() for value in lengths):
        raise AssertionError("最终有效KV长度或资源归还不符")
    if path.startswith("graph-"):
        if (result["graph"] != {"calls": 63, "fallbacks": 0, "replays": 63}
                or result["capture_delta"] or result["captures"] != 1):
            raise AssertionError("Graph未按64输出命中63次，或发生回退/重捕获")
        if path == "graph-overlap":
            stats = result["pipeline"]
            if (stats.get("device_tokens") != 63 or stats.get("committed_tokens") != 63
                    or stats.get("discarded_tokens", 0) or result["max_inflight"] != 2):
                raise AssertionError("流水实际发射、提交或双轮在途计数不符")


def difference(target, draft, item, path, expected, result):
    actual = result["output_ids"]
    position = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None)
    if position is None:
        return None
    evidence = dict(position=position, reference_token=expected[position], actual_token=actual[position],
                    common_prefix=expected[:position], reference_sha256=digest(expected), actual_sha256=digest(actual))
    evidence["reference_top2"] = reference(target, item["gpu"], inspect_at=position)["candidate"]
    if path.startswith("spec-"):
        def observe(event, targets, drafts, logits):
            offset = position - event["output_start"]
            if 0 <= offset < len(event["committed"]):
                evidence["verification_event"] = event.copy()
                evidence["verification_top2"] = top2(logits[0, offset])
        generate_speculative(target, draft, item["gpu"], draft_tokens=int(path[-1]),
                             max_new_tokens=OUTPUT_TOKENS, ignore_eos=True, observer=observe)
    else:
        # CUDA eager只作同历史诊断，不冒充原Graph/流水的实际logits。
        caches = make_caches(target, item["gpu"].shape[1] + OUTPUT_TOKENS)
        try:
            logits = forward_tokens(target, item["gpu"], caches)
            for token in expected[:max(0, position - 1)]:
                forward_tokens(target, item["gpu"].new_tensor([[token]]), caches)
            if position:
                current = item["gpu"].new_tensor([[expected[position - 1]]])
                positions = current.new_tensor([[caches[0].length]])
                logits = model_forward(target, current, positions, caches, is_prefill=False, cuda_decode=True)
            evidence["same_history_cuda_eager_top2"] = top2(logits[0, -1])
        finally:
            for cache in caches:
                cache.release()
    return evidence


def evaluate(target, draft, scheduler, item, path, phase, index, report, expected, diagnostic=None):
    torch.cuda.synchronize()
    started = time.perf_counter()
    if path == "reference":
        result = reference(target, item["gpu"])
    elif path.startswith("spec-"):
        result = generate_speculative(target, draft, item["gpu"], draft_tokens=int(path[-1]),
                                      max_new_tokens=OUTPUT_TOKENS, ignore_eos=True, diagnostic=diagnostic)
    else:
        result = graph_request(scheduler, item["gpu"], path == "graph-overlap")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    row = dict(input=item["id"], path=path, phase=phase, repeat=index, wall_ms=wall_ms,
               output_ids=result["output_ids"], output_sha256=digest(result["output_ids"]),
               final_kv_lengths=result["final_kv_lengths"], recovered=result["recovered"])
    report["runs"].append(row)
    check_result(path, result, item["gpu"].shape[1])
    if path.startswith("spec-"):
        row["speculative"] = speculative_counts(result, int(path[-1]))
    else:
        row.update({key: result[key] for key in ("graph", "captures", "pipeline", "max_inflight") if key in result})
    if diagnostic:
        row["stages"] = diagnostic.report()
    if expected is not None:
        row["difference"] = difference(target, draft, item, path, expected, result)
        if row["difference"] is not None:
            print(f"[分歧] 输入={item['id']} 路径={path} 证据={row['difference']}")
            raise RuntimeError("完整输出不一致；该输入不生成性能结论，已保留原始样本与首处分歧")
    if phase == "warmup":
        print(f"[检查] 输入={item['id']} 路径={path} 输出=64 回收={row['recovered']} "
              f"Graph={row.get('graph')} 投机计数={row.get('speculative', {}).get('totals')}")
    return result["output_ids"]


@torch.inference_mode()
def run(args, report):
    target, draft, tokenizer = load_pair()
    selected = PROMPTS if args.mode == "paired" else [next(p for p in PROMPTS if p[0] == args.prompt)]
    inputs = []
    for name, text in selected:
        ids = encode_prompt(tokenizer, text)
        tokens = ids[0].tolist()
        inputs.append(dict(id=name, text=text, gpu=ids, token_ids=tokens, input_tokens=len(tokens), sha256=digest(tokens)))
    report["inputs"] = [{key: value for key, value in item.items() if key != "gpu"} for item in inputs]
    report["resident_allocated_before_graph_bytes"] = torch.cuda.memory_allocated()
    scheduler, report["graph_initialization"] = initialize_graph(target, inputs)
    print(f"[初始化] 两模型常驻；共用B1图池={report['graph_initialization']}")
    for item in inputs:
        expected = evaluate(target, draft, scheduler, item, "reference", "warmup", 0, report, None)
        if args.mode == "diagnose":
            for k in (2, 4):
                path = f"spec-k{k}"
                evaluate(target, draft, scheduler, item, path, "warmup", 0, report, expected)
                evaluate(target, draft, scheduler, item, path, "diagnostic", 0, report, expected, Stages())
                print(f"[阶段] 输入={item['id']} K={k} {report['runs'][-1]['stages']}")
            continue
        for path in PATHS[1:]:
            evaluate(target, draft, scheduler, item, path, "warmup", 0, report, expected)
        orders = [PATHS] if args.mode == "smoke" else [PATHS, PATHS[::-1], PATHS[2:] + PATHS[:2]]
        for repeat, order in enumerate(orders):
            report["orders"].append(dict(input=item["id"], repeat=repeat, paths=list(order)))
            for path in order:
                evaluate(target, draft, scheduler, item, path, "sample", repeat, report, expected)
        summary = {}
        for path in PATHS:
            samples = [row["wall_ms"] for row in report["runs"]
                       if row["input"] == item["id"] and row["path"] == path and row["phase"] == "sample"]
            median = statistics.median(samples)
            summary[path] = dict(samples_ms=samples, median_ms=median, output_tokens_per_s=64000 / median)
            print(f"[配对] 输入={item['id']} Token={item['input_tokens']} 路径={path} 样本={samples} 中位={median:.3f}ms")
        best = min(("graph-sync", "graph-overlap"), key=lambda path: summary[path]["median_ms"])
        report["summaries"][item["id"]] = dict(paths=summary, faster_graph_path=best,
            speculative_ratios={path: {base: summary[base]["median_ms"] / summary[path]["median_ms"]
                                      for base in ("reference", best)} for path in ("spec-k2", "spec-k4")})
    report["graph_final"] = graph_counts(scheduler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paired", "diagnose"), default="smoke")
    parser.add_argument("--prompt", choices=[name for name, text in PROMPTS], default="cpp")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = dict(run_id=args.run_id, mode=args.mode, status="running", runs=[], orders=[], summaries={},
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  tracked_dirty=bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip()),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                   transformers=version("transformers"), gpu=torch.cuda.get_device_name(0),
                                   total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
                  config=dict(target=[MODEL_ID, MODEL_REVISION], draft=[DRAFT_ID, DRAFT_REVISION],
                              batch=1, dtype="bfloat16", output_tokens=64, ignore_eos=True,
                              template=dict(enable_thinking=False, add_generation_prompt=True, tokenizer="target"),
                              warmups=1, repeats=3 if args.mode == "paired" else int(args.mode == "smoke"),
                              resident_models="target+draft", shared_graph_pool_all_paths=True,
                              request_pools="reference:target; spec:target+draft; graph:常驻池复用",
                              timing="整请求含Prefill/宿主Token/资源归还与末尾等待；准备/捕获/比较在计时外",
                              stages="独立诊断含已有取词等待；宿主与GPU流区间可重叠，不相加"))
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[完成] {args.run_id} 状态={report['status']} JSON={args.output}")


if __name__ == "__main__":
    main()
