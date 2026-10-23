"""闭环离线负载实验；复用生产调度，不提供服务SLO或异步收益结论。"""

import argparse
from collections import Counter, deque
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class Meter:
    """仅保留在途引用、固定窗口、三个完成样本和滚动摘要。"""

    def __init__(self, seconds):
        self.start = time.perf_counter()
        self.seconds = seconds
        self.live = {}
        self.windows = [Counter() for _ in range(3 if seconds else 1)]
        self.drain = Counter()
        self.sequence = hashlib.sha256()
        self.samples = []
        self.submitted = self.completed = self.rounds = 0
        self.completed_outputs = 0
        self.last_ready = None

    def bucket(self, now):
        elapsed = now - self.start
        if self.seconds and elapsed >= self.seconds:
            return self.drain
        index = min(2, int(elapsed * 3 / self.seconds)) if self.seconds else 0
        return self.windows[index]

    def ready(self, request, now):
        self.last_ready = now
        meta = self.live[request.request_id]
        bucket = self.bucket(now)
        bucket["tokens_ready"] += 1
        if meta["last"] is None:
            bucket["first_tokens"] += 1
            bucket["first_ready_ms_sum"] += (now - meta["arrival"]) * 1000
        else:
            gap = (now - meta["last"]) * 1000
            bucket["token_gaps"] += 1
            bucket["token_gap_ms_sum"] += gap
            bucket["token_gap_ms_max"] = max(bucket["token_gap_ms_max"], gap)
        meta["last"] = now

    def finish(self, request, now, observed):
        meta = self.live.pop(request.request_id)
        assert meta["request"] is request and request.status == "finished"
        assert len(request.output_ids) == request.max_new_tokens and request.stop_reason == "达到生成上限"
        if observed:
            assert meta["final_kv"] == meta["template"]["input_tokens"] + request.max_new_tokens - 1
        self.completed += 1
        bucket = self.bucket(now)
        bucket["completed"] += 1
        self.completed_outputs += len(request.output_ids)
        if observed:
            bucket["final_kv_min"] = min(bucket.get("final_kv_min", meta["final_kv"]), meta["final_kv"])
            bucket["final_kv_max"] = max(bucket["final_kv_max"], meta["final_kv"])
        bucket["request_step_return_ms_sum"] += (now - meta["arrival"]) * 1000
        self.sequence.update(json.dumps((meta["template"]["id"], request.output_ids)).encode())
        if len(self.samples) < 3:
            self.samples.append({"request_id": request.request_id, "template": meta["template"]["id"],
                                 "outputs": len(request.output_ids), "sha256": digest(request.output_ids),
                                 "final_kv": meta.get("final_kv")})


def scheduler_type():
    import torch
    from ampere_kv.scheduler import Scheduler

    class ObservedScheduler(Scheduler):
        meter = None
        tracing = False

        def trace(self, label):
            return torch.cuda.nvtx.range(label) if self.tracing else nullcontext()

        def step(self):
            self.batches, self.prefilled, self.retired = [], False, 0
            with self.trace(f"step:round={self.meter.rounds + 1},running_before={len(self.running)}"):
                return super().step()

        def _advance(self, request, *, prefill):
            self.prefilled |= prefill
            if not prefill:
                self.batches.append((1, False))
            stage = "prefill_request" if prefill else "decode_single"
            with self.trace(f"{stage}:history={request.caches[0].length}"):
                return super()._advance(request, prefill=prefill)

        def _decode_batch(self, group, graph=None):
            self.batches.append((len(group), graph is not None))
            stage = "decode_graph" if graph is not None else "decode_batch"
            with self.trace(f"{stage}:batch={len(group)}"):
                return super()._decode_batch(group, graph)

        def _commit(self, request, token):
            self.meter.ready(request, time.perf_counter())
            with self.trace("commit"):
                return super()._commit(request, token)

        def _retire(self, request):
            self.retired += 1
            self.meter.live[request.request_id]["final_kv"] = request.caches[0].length
            with self.trace("retire"):
                return super()._retire(request)

        def _stop_round(self, now):
            with self.trace("stop_round"):
                return super()._stop_round(now)

        def _admit(self):
            with self.trace("admit"):
                return super()._admit()

    return ObservedScheduler


def run(scheduler, templates, batch, requests=0, seconds=0, profile=None, observed=True, pipeline_mode="a"):
    import torch

    torch.cuda.synchronize()  # 样本起点排空预热；不在每Token增加同步。
    torch.cuda.reset_peak_memory_stats()
    meter = Meter(seconds)
    scheduler.meter = meter
    scheduler.tracing = False
    pipeline = scheduler.decode_pipeline if pipeline_mode != "a" else None
    if pipeline:
        assert pipeline.pending is None
        pipeline.overlap = pipeline_mode == "c"
        pipeline.stats["max_inflight"] = 0
    pipeline_before = pipeline.stats.copy() if pipeline else Counter()
    captures = scheduler.graph_captures
    before = (scheduler.graph_calls, scheduler.graph_fallbacks,
              {str(b): g.replays for b, g in (scheduler.graphs or {}).items()})
    memory = deque(maxlen=64)
    profile_start = profile_end = None
    profile_counts = Counter()

    def snapshot(now, phase):
        memory.append({"elapsed_s": now - meter.start, "phase": phase,
                       "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                       "inflight": len(meter.live), "waiting": len(scheduler.waiting),
                       "running": len(scheduler.running), "prefilling": int(scheduler.prefilling is not None)})

    def fill():
        while len(meter.live) < batch:
            now = time.perf_counter()
            if (seconds and now - meter.start >= seconds) or (not seconds and meter.submitted >= requests):
                break
            template = templates[meter.submitted % len(templates)]
            serial = getattr(scheduler, "workload_serial", 0)
            scheduler.workload_serial = serial + 1
            request = scheduler.submit(f"r{serial}", template["gpu"], template["outputs"], ignore_eos=True)
            meter.live[request.request_id] = {"request": request, "template": template, "arrival": now, "last": None}
            meter.submitted += 1

    snapshot(meter.start, "idle_before")
    next_memory = meter.start + 30
    fill()
    try:
        while meter.live:
            number = meter.rounds + 1
            if profile and number == profile[0]:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                scheduler.tracing = True
                profile_start = number
            started = time.perf_counter()
            bucket = meter.bucket(started)
            bucket["inflight_max"] = max(bucket["inflight_max"], len(meter.live))
            bucket["waiting_max"] = max(bucket["waiting_max"], len(scheduler.waiting))
            bucket["waiting_sum"] += len(scheduler.waiting)
            bucket["running_max"] = max(bucket["running_max"], len(scheduler.running))
            if pipeline:
                scheduler.batches, scheduler.prefilled, scheduler.retired = [], False, 0
                pipeline.tracing = scheduler.tracing
                finished, sizes = pipeline.advance()
                if not pipeline.used_fallback:
                    scheduler.batches = pipeline.last_batches
            else:
                finished, sizes = scheduler.step()
            ended = time.perf_counter()
            meter.rounds += 1
            bucket["rounds"] += 1
            elapsed = (ended - started) * 1000
            bucket["step_ms_sum"] += elapsed
            bucket["step_ms_max"] = max(bucket["step_ms_max"], elapsed)
            if observed:
                assert sizes == [size for size, graph in scheduler.batches]
                kind = "pure_decode" if sizes and not scheduler.prefilled and not scheduler.retired else "other"
                bucket[kind + "_rounds"] += 1
                bucket[kind + "_ms_sum"] += elapsed
                bucket["prefill_rounds"] += scheduler.prefilled
                bucket["retire_rounds"] += bool(scheduler.retired)
                for size, graph in scheduler.batches:
                    key = f"{'graph' if graph else 'eager'}_b{size}_calls"
                    bucket[key] += 1
                    if scheduler.tracing:
                        profile_counts[key] += 1
            for request in finished:
                meter.finish(request, ended, observed)
            if profile and number == profile[0] + profile[1] - 1:
                torch.cuda.synchronize()  # 仅选定采集窗口末尾排空一次。
                torch.cuda.profiler.stop()
                scheduler.tracing = False
                profile_end = number
            fill()  # 只在step返回后补入，waiting也占在途额度。
            if ended >= next_memory:
                sampled = time.perf_counter()
                snapshot(sampled, "active")
                next_memory = sampled + 30
    finally:
        if scheduler.tracing:
            torch.cuda.profiler.stop()
            scheduler.tracing = False
    torch.cuda.synchronize()
    ended = time.perf_counter()
    snapshot(ended, "idle_after")
    assert meter.completed == meter.submitted and scheduler.reserved_blocks == 0
    assert not scheduler.waiting and not scheduler.running and scheduler.prefilling is None
    assert all(s._pool.num_free_blocks == scheduler.blocks_per_layer for s in scheduler.storages)
    assert scheduler.graph_captures == captures
    assert pipeline is None or pipeline.pending is None
    if pipeline:
        delta = pipeline.stats - pipeline_before
        assert delta["device_tokens"] == delta["committed_tokens"] + delta["discarded_tokens"]
    if profile:
        assert profile_end is not None, "请求结束前未覆盖所选采集窗口"
    output_tokens = sum(w["tokens_ready"] for w in meter.windows) + meter.drain["tokens_ready"]
    if observed:
        assert output_tokens == meter.completed_outputs
        buckets = [*meter.windows, meter.drain]
        graph_calls = sum(v for w in buckets for k, v in w.items() if k.startswith("graph_b") and k.endswith("_calls"))
        assert graph_calls == scheduler.graph_calls - before[0]
        if scheduler.graphs:
            eager_calls = sum(v for w in buckets for k, v in w.items() if k.startswith("eager_b") and k.endswith("_calls"))
            assert eager_calls == scheduler.graph_fallbacks - before[1]
    observation = seconds or ended - meter.start
    windows = []
    for index, bucket in enumerate(meter.windows):
        span = seconds / 3 if seconds else observation
        windows.append({"phase": ("early", "middle", "late")[index] if seconds else "all",
                        "seconds": span, "output_tokens_per_s": bucket["tokens_ready"] / span,
                        "counts": dict(bucket)})
    return {"observation_s": observation, "total_s": ended - meter.start,
            "pipeline_mode": pipeline_mode,
            "last_output_s": meter.last_ready - meter.start if meter.last_ready else None,
            "completion_after_last_output_ms": (ended - meter.last_ready) * 1000 if meter.last_ready else None,
            "pipeline": {k: pipeline.stats[k] - pipeline_before[k] for k in pipeline.stats
                         if k != "max_inflight"} | {"max_inflight": pipeline.stats["max_inflight"]} if pipeline else {},
            "decode_forward_calls": sum(v for w in [*meter.windows, meter.drain] for k, v in w.items()
                                        if k.endswith("_calls") and k.startswith(("eager_b", "graph_b"))),
            "prefill_forward_calls": sum(w["prefill_rounds"] for w in [*meter.windows, meter.drain]),
            "drain_s": max(0, ended - meter.start - seconds) if seconds else 0,
            "submitted": meter.submitted, "completed": meter.completed, "rounds": meter.rounds,
            "output_tokens": output_tokens, "completed_output_tokens": meter.completed_outputs,
            "windows": windows, "drain": dict(meter.drain), "output_sha256": meter.sequence.hexdigest(),
            "samples": meter.samples, "memory": list(memory), "peak_allocated": torch.cuda.max_memory_allocated(),
            "idle_allocated_delta": memory[-1]["allocated"] - memory[0]["allocated"],
            "recovery": {"reserved_blocks": scheduler.reserved_blocks, "layers": len(scheduler.storages),
                         "all_blocks_free": True},
            "graph": {"calls": scheduler.graph_calls - before[0], "fallbacks": scheduler.graph_fallbacks - before[1],
                      "captures": captures, "replays": {str(b): g.replays - before[2][str(b)]
                                                        for b, g in (scheduler.graphs or {}).items()}},
            "profile_window": {"start_round": profile_start, "end_round": profile_end,
                               "batch_calls": dict(profile_counts)} if profile else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "steady", "stream"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=4)
    parser.add_argument("--output-tokens", type=int, choices=(32, 128, 256, 512), default=256)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--pipeline", choices=("a", "b", "c", "abc"), default="a")
    parser.add_argument("--profile-rounds", nargs=2, type=int, metavar=("START", "COUNT"))
    args = parser.parse_args()
    if args.pipeline != "a" and (not args.cuda_graph or args.batch not in (1, 4)):
        parser.error("B/C仅支持显式Graph B1/B4")
    if args.pipeline == "abc" and (args.mode != "steady" or args.profile_rounds):
        parser.error("ABC配对仅用于无采集steady；诊断请选择单独C")
    repo = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo) or args.output.exists():
        parser.error("结果须写到仓库外的新文件，不覆盖旧证据")
    if not 1 <= args.seconds <= 1200:
        parser.error("持续时间须为1至1200秒，截止后正常排空")
    if args.profile_rounds and (args.mode == "smoke" or min(args.profile_rounds) < 1
                               or args.profile_rounds[1] > 64 or (args.mode == "stream" and args.seconds > 60)):
        parser.error("只采预热后正轮次窗口，最多64轮；stream诊断最多60秒")
    import torch
    from ampere_kv.bench_external import load_inputs
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION, load_model_and_tokenizer
    from ampere_kv.scheduler import Scheduler

    prompts, file_hash, source_hash = load_inputs(args.input, 4, 512, False)
    specs = ([(128, 3), (512, 5), (128, 4), (512, 6)] if args.mode == "smoke" else
             [(128, 32), (512, 128), (128, 256), (512, 32), (128, 128), (512, 256)]
             if args.mode == "stream" else [(512, args.output_tokens)] * args.batch)
    templates = []
    for index, (length, outputs) in enumerate(specs):
        ids = prompts[index % 4][:length]
        templates.append({"id": f"t{index}", "source_index": index % 4, "input_tokens": length,
                          "outputs": outputs, "sha256": digest(ids), "ids": ids})
    model, _ = load_model_and_tokenizer()
    blocks = args.batch * max((t["input_tokens"] + t["outputs"] + 16) // 16 for t in templates)
    needed = blocks * 16 * len(model.model.layers) * 2 * model.config.num_key_value_heads * model.config.head_dim * 2
    free, _ = torch.cuda.mem_get_info()
    result = {"run_id": args.run_id, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "model": MODEL_ID, "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "config": vars(args) | {"input": str(args.input), "output": str(args.output)},
              "input_file_sha256": file_hash, "source_ids_sha256": source_hash,
              "template_sequence_sha256": digest([(t["sha256"], t["outputs"]) for t in templates]),
              "templates": [{k: v for k, v in t.items() if k != "ids"} for t in templates],
              "blocks_per_layer": blocks, "pool_bytes": needed, "free_before_pool": free, "headroom_bytes": 1024**3,
              "timing": "profile_diagnostic_not_formal" if args.profile_rounds else "closed_loop_wallclock_with_driver",
              "boundary": "闭环离线/ignoreEOS/BF16/V1/mixed关闭；驱动开销包含在内，非服务SLO或GPU纯耗时"}
    if needed + 1024**3 > free:
        result.update(status="skipped_memory", reason="估算KV池加1GiB余量超过当前空闲显存，不保证通过预检即不会OOM")
    else:
        for template in templates:
            template["gpu"] = torch.tensor([template.pop("ids")], dtype=torch.long, device="cuda")
        try:
            init_start = time.perf_counter()
            init_allocated = torch.cuda.memory_allocated()
            scheduler = scheduler_type()(model, blocks, max_batch=args.batch, mixed=False)
            if args.cuda_graph:
                capacity = max(t["input_tokens"] + t["outputs"] for t in templates)
                scheduler.enable_graphs((capacity + 15) // 16 * 16)
            if args.pipeline != "a":
                scheduler.enable_pipeline(overlap=args.pipeline == "c")
            torch.cuda.synchronize()
            result["initialization"] = {"wall_ms": (time.perf_counter() - init_start) * 1000,
                                        "allocated_delta": torch.cuda.memory_allocated() - init_allocated,
                                        "includes": "KV池、Graph及可选流水槽；不计入测量"}
            warm = [t | {"outputs": 8} for t in templates]
            modes = "abc" if args.pipeline == "abc" else args.pipeline
            for mode in modes:
                run(scheduler, warm, args.batch, requests=args.batch, pipeline_mode=mode)
            samples = []
            for repeat in range(1 if args.mode != "steady" or args.profile_rounds else args.repeats):
                for mode in modes[::-1] if repeat % 2 else modes:
                    samples.append(run(scheduler, templates, args.batch,
                                       requests=2 * args.batch if args.mode == "smoke" else args.batch,
                                       seconds=args.seconds if args.mode == "stream" else 0,
                                       profile=args.profile_rounds, pipeline_mode=mode))
            if args.pipeline == "abc":
                assert len({s["output_sha256"] for s in samples}) == 1, "ABC完整输出摘要不同，不报告加速"
            if args.mode == "smoke":
                reference = Scheduler(model, blocks, max_batch=args.batch, mixed=False)
                expected = run(reference, templates, args.batch, requests=2 * args.batch, observed=False)
                assert samples[0]["output_sha256"] == expected["output_sha256"], "短序列与原Scheduler同调度方式不一致"
                assert samples[0]["submitted"] > args.batch
                result["smoke_reference_match"] = True
            result.update(status="passed", samples=samples, profile_diagnostic=bool(args.profile_rounds))
            result["medians_by_pipeline_ms"] = {m: statistics.median(s["total_s"] * 1000 for s in samples
                                                                       if s["pipeline_mode"] == m) for m in modes}
            result["median_total_ms"] = result["medians_by_pipeline_ms"][modes] if len(modes) == 1 else None
        except torch.cuda.OutOfMemoryError as exc:
            result.update(status="oom", reason=str(exc))  # 退出本进程，不降批量重试冒充原配置。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[结果] {args.run_id} 状态={result['status']}，JSON={args.output}")
    for sample in result.get("samples", []):
        print(f"[样本] 路径={sample['pipeline_mode']}，完成={sample['completed']}，轮次={sample['rounds']}，总秒={sample['total_s']:.3f}，排空秒={sample['drain_s']:.3f}，Graph={sample['graph']}")
        if sample["pipeline"]:
            print(f"[流水] {sample['pipeline']}，Decode前向={sample['decode_forward_calls']}，最后输出秒={sample['last_output_s']:.6f}，输出后收尾ms={sample['completion_after_last_output_ms']:.3f}")
        for window in sample["windows"]:
            print(f"[窗口] {window['phase']} 输出吞吐={window['output_tokens_per_s']:.3f} Token/s，统计={window['counts']}")
    print("[边界] Token按commit入口就绪、轮次按轮首分窗；分类可重叠，profile结果不作正式性能。")
    return int(result["status"] == "oom")


if __name__ == "__main__":
    raise SystemExit(main())
