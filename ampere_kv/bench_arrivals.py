"""单线程预定到达负载；闭环校准与正式到达分开，计划时间不随引擎忙碌推迟。"""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


SCENARIOS = ("low", "steady", "overload-probe", "burst")
INPUT_TOKENS, OUTPUT_TOKENS, BLOCK_SIZE = 512, 32, 16


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def arrival_table(rate, seconds, burst=False):
    count = math.ceil(rate * seconds)
    if count > 256:
        raise ValueError(f"预定{count}请求超过256上限；请缩短--seconds或降低--rate，不截断到达表")
    return [dict(id=f"r{i}", arrival_s=(i // 8 * 8 if burst else i) / rate,
                 template=i % 4, output_limit=OUTPUT_TOKENS) for i in range(count)]


def describe(values):
    """P95固定为线性插值：排序后位置(n-1)*0.95；空样本返回None。"""
    values = sorted(values)
    if not values:
        return dict(n=0, median=None, p95=None, maximum=None)
    position = (len(values) - 1) * 0.95
    lo = math.floor(position)
    p95 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)
    return dict(n=len(values), median=statistics.median(values), p95=p95, maximum=values[-1])


class Observation:
    def __init__(self, events, clock=time.perf_counter):
        self.clock, self.start = clock, clock()
        self.rows = {e["id"]: dict(e, submit_s=None, admit_s=None, retire_s=None, complete_s=None,
                                   token_times_s=[], status="planned", output_sha256=None) for e in events}
        self.live, self.snapshots, self.batches = {}, [], Counter()
        self.cutting = False

    def now(self):
        return self.clock() - self.start

    def submit(self, scheduler, templates, event):
        row = self.rows[event["id"]]
        row["submit_s"], row["status"] = self.now(), "submitted"
        self.live[event["id"]] = scheduler.submit(event["id"], templates[event["template"]],
                                                 event["output_limit"], ignore_eos=True)

    def snapshot(self, scheduler):
        now = self.now()
        allocated = [scheduler.blocks_per_layer - s._pool.num_free_blocks for s in scheduler.storages]
        due = sum(r["arrival_s"] <= now and r["submit_s"] is None for r in self.rows.values())
        self.snapshots.append(dict(at_s=now, waiting=len(scheduler.waiting), running=len(scheduler.running),
                                   prefilling=int(scheduler.prefilling is not None), live=len(self.live),
                                   arrived_not_submitted=due, reserved_blocks_per_layer=scheduler.reserved_blocks,
                                   allocated_blocks_per_layer_min=min(allocated),
                                   allocated_blocks_per_layer_max=max(allocated)))


def scheduler_type():
    from ampere_kv.scheduler import Scheduler

    class ObservedScheduler(Scheduler):
        def _admit(self):
            super()._admit()
            if self.prefilling is not None:
                row = self.observation.rows[self.prefilling.request_id]
                if row["admit_s"] is None:
                    row["admit_s"] = self.observation.now()

        def _commit(self, request, token):
            self.observation.rows[request.request_id]["token_times_s"].append(self.observation.now())
            super()._commit(request, token)

        def _advance(self, request, *, prefill):
            if not prefill:
                self.observation.batches[("eager", 1)] += 1
            return super()._advance(request, prefill=prefill)

        def _decode_batch(self, group, graph=None):
            self.observation.batches[("graph" if graph is not None else "eager", len(group))] += 1
            return super()._decode_batch(group, graph)

        def _retire(self, request):
            observation = self.observation
            row = observation.rows[request.request_id]
            normal = not observation.cutting
            lengths = [cache.length for cache in request.caches]
            row["final_kv_min"] = min(lengths, default=0)
            row["final_kv_max"] = max(lengths, default=0)
            assert len(request.output_ids) == len(row["token_times_s"])
            assert all(0 <= token < self.model.config.vocab_size for token in request.output_ids)
            if normal:
                assert len(request.output_ids) == row["output_limit"] and request.stop_reason == "达到生成上限"
                assert row["final_kv_min"] == row["final_kv_max"] == INPUT_TOKENS + row["output_limit"] - 1
                row["complete_s"] = row["token_times_s"][-1]
            row["output_sha256"], row["stop_reason"] = digest(request.output_ids), request.stop_reason
            super()._retire(request)
            row["retire_s"] = observation.now()
            row["status"] = "completed" if normal else "measurement_cutoff_cancelled"
            observation.live.pop(request.request_id)

    return ObservedScheduler


def recovered(scheduler):
    return (not scheduler.waiting and not scheduler.running and scheduler.prefilling is None
            and scheduler.reserved_blocks == 0
            and all(s._pool.num_free_blocks == scheduler.blocks_per_layer for s in scheduler.storages))


def calibration(scheduler, templates, count, synchronize):
    """仅校准使用完成后补充，最多4个在途；正式场景不调用此驱动。"""
    synchronize()
    events = [dict(id=f"c{i}", arrival_s=0.0, template=i % 4, output_limit=OUTPUT_TOKENS) for i in range(count)]
    observation = Observation(events)
    scheduler.observation = observation
    index = 0
    while index < count or observation.live:
        while index < count and len(observation.live) < 4:
            event = events[index]
            observation.rows[event["id"]]["arrival_s"] = observation.now()
            observation.submit(scheduler, templates, event)
            index += 1
        scheduler.step()
    elapsed = observation.now()
    assert recovered(scheduler)
    return dict(kind="closed_loop_reference", requests=count, completed=count, seconds=elapsed,
                requests_per_s=count / elapsed, max_inflight=4, recovered=True,
                output_sha256=digest([(r["template"], r["output_sha256"]) for r in observation.rows.values()]))


def summarize(scheduler, observation, seconds, elapsed, before, unfinished):
    rows = list(observation.rows.values())
    completed = [r for r in rows if r["status"] == "completed"]
    admitted = [r for r in rows if r["admit_s"] is not None]
    submitted = [r for r in rows if r["submit_s"] is not None]
    ready = [r for r in rows if r["token_times_s"]]
    for row in rows:
        times = [row["arrival_s"], row["submit_s"], row["admit_s"], *row["token_times_s"], row["retire_s"]]
        times = [t for t in times if t is not None]
        assert all(a <= b for a, b in zip(times, times[1:])), row["id"]
    metrics = dict(driver_delay_ms=describe([(r["submit_s"] - r["arrival_s"]) * 1000 for r in submitted]),
                   engine_wait_ms=describe([(r["admit_s"] - r["submit_s"]) * 1000 for r in admitted]),
                   admission_wait_ms=describe([(r["admit_s"] - r["arrival_s"]) * 1000 for r in admitted]),
                   ttft_ms=describe([(r["token_times_s"][0] - r["arrival_s"]) * 1000 for r in ready]),
                   completion_ms=describe([(r["complete_s"] - r["arrival_s"]) * 1000 for r in completed]),
                   token_gap_ms=describe([(b - a) * 1000 for r in ready
                                          for a, b in zip(r["token_times_s"], r["token_times_s"][1:])]))
    inside = lambda t: t is not None and 0 <= t < seconds
    window_completed = sum(inside(r["complete_s"]) for r in completed)
    window_tokens = sum(inside(t) for r in ready for t in r["token_times_s"])
    histogram = {str(n): sum(v for (kind, size), v in observation.batches.items() if size == n)
                 for n in sorted({size for kind, size in observation.batches})}
    assert sum(int(n) * count for n, count in histogram.items()) == sum(max(0, len(r["token_times_s"]) - 1) for r in rows)
    graph = dict(calls=scheduler.graph_calls - before[0], fallbacks=scheduler.graph_fallbacks - before[1],
                 captures=scheduler.graph_captures,
                 replays={str(b): g.replays - before[2][b] for b, g in (scheduler.graphs or {}).items()})
    assert graph["calls"] == sum(v for (kind, n), v in observation.batches.items() if kind == "graph")
    per_second = [dict(second=i, arrivals=0, submitted=0, admitted=0, completed=0) for i in range(math.floor(elapsed) + 1)]
    for row in rows:
        for name, key in (("arrivals", "arrival_s"), ("submitted", "submit_s"),
                          ("admitted", "admit_s"), ("completed", "complete_s")):
            if row[key] is not None:
                per_second[math.floor(row[key])][name] += 1
    cancelled = sum(r["status"] == "measurement_cutoff_cancelled" for r in rows)
    unsubmitted = sum(r["status"] == "not_submitted_cutoff" for r in rows)
    assert len(completed) + cancelled + unsubmitted == len(rows)
    assert recovered(scheduler)
    return dict(counts=dict(planned=len(rows), submitted=len(submitted), admitted=len(admitted),
                            completed=len(completed), unfinished=len(rows) - len(completed),
                            unfinished_at_cutoff=unfinished, cutoff_cancelled=cancelled, not_submitted=unsubmitted),
                metrics=metrics, arrival_window=dict(seconds=seconds, completed=window_completed,
                    completed_requests_per_s=window_completed / seconds, committed_tokens=window_tokens,
                    output_tokens_per_s=window_tokens / seconds),
                including_drain=dict(seconds=elapsed, drain_s=max(0, elapsed - seconds),
                    completed_requests_per_s=len(completed) / elapsed,
                    committed_output_tokens_per_s=sum(len(r["token_times_s"]) for r in rows) / elapsed,
                    completed_request_output_tokens_per_s=sum(len(r["token_times_s"]) for r in completed) / elapsed),
                decode_batch_histogram=histogram, decode_paths={f"{kind}_b{n}": v for (kind, n), v in observation.batches.items()},
                graph=graph, recovered=True, per_second_events=per_second, state_observations=observation.snapshots,
                requests=rows, output_sha256=digest([(r["id"], r["output_sha256"], r["status"]) for r in rows]))


def run_arrivals(scheduler, templates, events, seconds, drain_seconds, synchronize,
                 clock=time.perf_counter, sleep=time.sleep):
    synchronize()
    observation = Observation(events, clock)
    scheduler.observation = observation
    before = (scheduler.graph_calls, scheduler.graph_fallbacks,
              {b: g.replays for b, g in (scheduler.graphs or {}).items()})
    index, next_sample, unfinished = 0, 0.0, 0
    while True:
        now = observation.now()
        if now >= seconds + drain_seconds and (observation.live or index < len(events)):
            unfinished = len(observation.live) + len(events) - index
            observation.snapshot(scheduler)
            observation.cutting = True
            for event in events[index:]:
                observation.rows[event["id"]]["status"] = "not_submitted_cutoff"
            for request in list(observation.live.values()):
                scheduler.cancel(request)
            while observation.live:
                scheduler.step()
            break
        # 每轮补交全部已到达事件；actual submit逐项取时，原arrival_s始终不变。
        while index < len(events) and events[index]["arrival_s"] <= observation.now():
            observation.submit(scheduler, templates, events[index])
            index += 1
        now = observation.now()
        if now >= seconds + drain_seconds and observation.live:
            continue
        if now >= next_sample:
            observation.snapshot(scheduler)
            next_sample = observation.now() + 1
        if not observation.live:
            if index == len(events) and now >= seconds:
                break
            target = events[index]["arrival_s"] if index < len(events) else seconds
            sleep(max(0, target - observation.now()))
        else:
            scheduler.step()
    observation.snapshot(scheduler)
    elapsed = observation.now()
    return summarize(scheduler, observation, seconds, elapsed, before, unfinished)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--rate", type=float, help="显式覆盖所有所选场景的请求/s；默认按校准mu固定分档")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--drain-seconds", type=float, default=120)
    parser.add_argument("--pool-requests", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--prefill-chunk-size", type=int, default=0, help="每轮Prefill块长；0为整段")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--graph-batches", type=int, nargs="+", help="显式捕获真实批量，例如1 2 3 4；默认沿用B1/B4")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.drain_seconds)
            or args.drain_seconds < 0 or args.repeats < 1 or args.pool_requests < 8 or args.max_batch < 1
            or args.prefill_chunk_size < 0
            or (args.rate is not None and (not math.isfinite(args.rate) or args.rate <= 0))):
        parser.error("时长/重复/批量须为正，排空时长和Prefill块长非负，池预算至少8请求，手动速率须为有限正数")
    if args.output.resolve().is_relative_to(root):
        parser.error("结果JSON须写入仓库外目录")
    if args.graph_batches is not None and (not args.cuda_graph or any(
            not 1 <= b <= min(args.max_batch, 4) for b in args.graph_batches)):
        parser.error("--graph-batches须同时启用--cuda-graph，批量限1至min(max-batch,4)")
    if args.rate is not None:
        try:
            arrival_table(args.rate, args.seconds)
        except ValueError as error:
            parser.error(str(error))
    import torch
    from ampere_kv.bench_external import load_inputs
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION, load_model_and_tokenizer

    prompts, file_hash, input_hash = load_inputs(args.input, 4, INPUT_TOKENS, False)
    model, _ = load_model_and_tokenizer()
    if any(token >= model.config.vocab_size for ids in prompts for token in ids):
        parser.error("输入Token超出固定模型词表")
    templates = [torch.tensor([ids], device="cuda", dtype=torch.long) for ids in prompts]
    budget = (INPUT_TOKENS + OUTPUT_TOKENS + BLOCK_SIZE) // BLOCK_SIZE
    width = math.ceil((INPUT_TOKENS + OUTPUT_TOKENS) / BLOCK_SIZE) * BLOCK_SIZE
    with torch.inference_mode():
        started = time.perf_counter()
        scheduler = scheduler_type()(model, args.pool_requests * budget, max_batch=args.max_batch,
                                     prefill_chunk_size=args.prefill_chunk_size)
        capture = None
        if args.cuda_graph:
            torch.cuda.synchronize()
            before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            capture_started = time.perf_counter()
            scheduler.enable_graphs(width, batches=args.graph_batches)
            torch.cuda.synchronize()
            capture = dict(wall_ms=(time.perf_counter() - capture_started) * 1000,
                           allocated_delta_bytes=torch.cuda.memory_allocated() - before[0],
                           reserved_delta_bytes=torch.cuda.memory_reserved() - before[1])
        torch.cuda.synchronize()
        initialization_ms = (time.perf_counter() - started) * 1000
        print(f"[初始化] 总耗时={initialization_ms:.3f}ms，实际捕获={sorted(scheduler.graphs or {})}，捕获成本={capture}")
        print(f"[Prefill] 块长={scheduler.prefill_chunk_size}（0为整段），分离执行，Prefill不进Graph")
        warmup = calibration(scheduler, templates, 4, torch.cuda.synchronize)
        reference = calibration(scheduler, templates, 16, torch.cuda.synchronize)
        mu = reference["requests_per_s"]
        rates = dict(zip(SCENARIOS, (0.25 * mu, 0.75 * mu, 1.25 * mu, 0.75 * mu)))
        if args.rate is not None:
            rates = dict.fromkeys(SCENARIOS, args.rate)
        print(f"[校准] 闭环16请求完成={reference['seconds']:.3f}s，mu={mu:.4f}请求/s；正式到达速率={rates}")
        try:
            tables = {name: arrival_table(rates[name], args.seconds, name == "burst")
                      for name in dict.fromkeys(args.scenarios)}
        except ValueError as error:
            parser.error(str(error))
        result = dict(run_id=args.run_id, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=version("transformers")), model=dict(id=MODEL_ID, revision=MODEL_REVISION),
            input_file_sha256=file_hash, input_ids_sha256=input_hash,
            config=dict(input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS, ignore_eos=True, greedy=True,
                weights_dtype="bf16", kv_dtype="bf16", decode_kernel="v1",
                prefill="chunked_sdpa" if scheduler.prefill_chunk_size else "whole_sdpa",
                prefill_chunk_size=scheduler.prefill_chunk_size,
                mixed=False, pipeline=False, structured=False, cuda_graph=args.cuda_graph, max_batch=args.max_batch,
                pool_requests=args.pool_requests, request_budget_blocks=budget, blocks_per_layer=scheduler.blocks_per_layer,
                block_size=BLOCK_SIZE, graph_capacity_tokens=width if args.cuda_graph else None,
                graph_batches=sorted(scheduler.graphs or {}),
                max_planned_requests=256, burst_group_size=8,
                arrival_seconds=args.seconds, max_drain_seconds=args.drain_seconds, repeats=args.repeats,
                rates=rates, rate_source="manual" if args.rate is not None else "fixed_fractions_of_closed_loop_mu",
                clock="perf_counter", percentiles="线性插值(n-1)*p，p=0.95", window="[0,arrival_seconds)",
                timestamps="相对起点秒；Token按commit入口，完成按最后Token，退休另记",
                observations="轮次边界约1秒一次，忙时允许缺口；截止同样在轮首观察",
                kv_observations="块池已分配编号与预留额度；物理KV预分配，不是torch allocated",
                metric_scopes="TTFT/间隔包含已输出的部分请求，完成延迟/完成请求率仅正常完成",
                timing="到达窗口起点至窗口结束及全部安全退休，含等待/空闲/排空；模型/输入/建池/捕获/预热在外"),
            initialization_ms=initialization_ms, graph_capture=capture, warmup=warmup, calibration=reference, scenes=[])
        for name, events in tables.items():
            scene = dict(name=name, rate=rates[name], arrivals=events, arrival_table_sha256=digest(events), samples=[])
            result["scenes"].append(scene)
            for repeat in range(args.repeats):
                sample = run_arrivals(scheduler, templates, events, args.seconds, args.drain_seconds, torch.cuda.synchronize)
                sample["repeat"] = repeat
                sample["same_outputs_as_first"] = not scene["samples"] or sample["output_sha256"] == scene["samples"][0]["output_sha256"]
                sample["same_decode_batches_as_first"] = not scene["samples"] or sample["decode_batch_histogram"] == scene["samples"][0]["decode_batch_histogram"]
                scene["samples"].append(sample)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[结果] {name} 第{repeat + 1}轮 计数={sample['counts']} TTFT={sample['metrics']['ttft_ms']} "
                      f"窗口={sample['arrival_window']} 含排空={sample['including_drain']} Graph={sample['graph']} "
                      f"输出摘要={sample['output_sha256']} 同表复跑输出一致={sample['same_outputs_as_first']} 回收={sample['recovered']}")
    print(f"[完成] {args.run_id}，JSON={args.output}；状态是实际边界观测，计划到达独立于完成")


if __name__ == "__main__":
    main()
