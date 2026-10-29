"""短动态负载与固定轮次插入诊断；CPU阶段记录通过NVTX标记关联GPU时间线。"""

import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from ampere_kv.bench_arrivals import (Observation, arrival_table, calibration, describe, digest,
                                     recovered, run_arrivals, scheduler_type)


RATE = 2.128913764807336


def diagnostic_type():
    class DiagnosticScheduler(scheduler_type()):
        def reset_recording(self, enabled):
            self.recording, self.round, self.records = enabled, 0, []

        @contextmanager
        def stage(self, kind, label, **fields):
            if not self.recording:
                yield None
                return
            name = f"LOAD03:r{self.round}:{label}"
            event = dict(id=len(self.records), kind=kind, round=self.round, nvtx=name,
                         start_s=self.observation.now(), **fields)
            self.records.append(event)
            try:
                with self.nvtx(name):
                    yield event
            finally:
                event["end_s"] = self.observation.now()

        def step(self):
            if not self.recording:
                return super().step()
            self.round += 1
            with self.stage("round", "round"):
                return super().step()

        def _decode_batch(self, group, graph=None):
            if not self.recording:
                return super()._decode_batch(group, graph)
            ids = [r.request_id for r in group]
            path = "graph" if graph is not None else "eager"
            with self.stage("decode", f"decode_B{len(group)}_{path}:{','.join(ids)}",
                            requests=ids, batch=len(group), graph=graph is not None):
                return super()._decode_batch(group, graph)

        def _advance(self, request, *, prefill):
            if not self.recording:
                return super()._advance(request, prefill=prefill)
            count = request.input_ids.shape[1] - request.caches[0].length if prefill else 1
            kind = "prefill" if prefill else "decode"
            with self.stage(kind, f"prefill_{count}:{request.request_id}" if prefill else "decode_B1_eager",
                            requests=[request.request_id], tokens=count):
                return super()._advance(request, prefill=prefill)

        def _commit(self, request, token):
            if not self.recording:
                return super()._commit(request, token)
            index = len(request.output_ids)
            with self.stage("commit", f"commit:{request.request_id}:o{index}",
                            request=request.request_id, output=index) as event:
                super()._commit(request, token)
                event["token_s"] = self.observation.rows[request.request_id]["token_times_s"][-1]

    return DiagnosticScheduler


def gap_analysis(rows, records, start_s=0, requests=None):
    """只算CPU区间关联；父子阶段不相加，GPU并集另从Nsight计算。"""
    commits = {(e["request"], e["output"]): e for e in records if e["kind"] == "commit"}
    prefills = [e for e in records if e["kind"] == "prefill"]
    gaps = []
    for row in rows:
        if requests is not None and row["id"] not in requests:
            continue
        times = row["token_times_s"]
        for index, (a, b) in enumerate(zip(times, times[1:])):
            if a < start_s:
                continue
            first, last = commits[row["id"], index], commits[row["id"], index + 1]
            assert first["token_s"] == a and last["token_s"] == b
            overlap = [e["id"] for e in prefills if max(a, e["start_s"]) < min(b, e["end_s"])]
            gaps.append(dict(request=row["id"], output_from=index, output_to=index + 1,
                             start_s=a, end_s=b, duration_ms=(b-a)*1000,
                             from_round=first["round"], to_round=last["round"], prefill_events=overlap))
    assert len(commits) == sum(t >= start_s for row in rows for t in row["token_times_s"])
    overlap = [g["duration_ms"] for g in gaps if g["prefill_events"]]
    clean = [g["duration_ms"] for g in gaps if not g["prefill_events"]]
    ordinary = [g for g in gaps if not g["prefill_events"]] or gaps
    median = describe([g["duration_ms"] for g in ordinary])["median"]
    selected = sorted(gaps, key=lambda g: g["duration_ms"], reverse=True)[:5]
    selected += sorted(ordinary, key=lambda g: abs(g["duration_ms"] - median))[:3]
    examples = []
    for gap in selected:
        events = [e["id"] for e in records if e["kind"] in ("round", "prefill", "decode")
                  and max(gap["start_s"], e["start_s"]) < min(gap["end_s"], e["end_s"])]
        examples.append(dict(gap, event_ids=events))
    return dict(with_prefill=describe(overlap), without_prefill=describe(clean), gaps=gaps, examples=examples)


@contextmanager
def capture_range(torch, enabled):
    if not enabled:
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        # 整段边界等待，确保关闭采集后再整理JSON和清理。
        try:
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()


def event(request_id, template):
    return dict(id=request_id, arrival_s=0.0, template=template, output_limit=32)


def insertion(scheduler, templates, input_hashes, insert, expected_state=None):
    """两边同样按序准备四请求；观测起点保留在同一perf_counter时间轴。"""
    scheduler.reset_recording(False)
    planned = [event(f"r{i}", i) for i in range(4)]
    observation = Observation(planned)
    scheduler.observation = observation
    for item in planned:
        observation.submit(scheduler, templates, item)
    while len(scheduler.running) < 4:
        scheduler.step()
    old = list(scheduler.running)
    state = [dict(id=r.request_id, input_sha256=input_hashes[i], output_ids=list(r.output_ids),
                  kv_lengths=[c.length for c in r.caches], remaining=32-len(r.output_ids))
             for i, r in enumerate(old)]
    if expected_state is not None:
        assert state == expected_state, "A/B旧请求共同起点不同，停止比较"
    assert all(r["remaining"] > 6 for r in state), "准备后的第4轮前已有请求将退出"
    start = observation.now()
    scheduler.reset_recording(True)
    while observation.live:
        if insert and scheduler.round == 3:
            item = event("r4", 0)
            item["arrival_s"] = observation.now()
            observation.rows["r4"] = dict(item, submit_s=None, admit_s=None, retire_s=None,
                complete_s=None, token_times_s=[], status="planned", output_sha256=None)
            observation.submit(scheduler, templates, item)
        scheduler.step()
    assert recovered(scheduler)
    rows, records = list(observation.rows.values()), scheduler.records
    analysis = gap_analysis(rows, records, start, {r.request_id for r in old})
    local = [g for g in analysis["gaps"] if g["from_round"] in (2, 3, 4, 5, 6)]
    if insert:
        prefill = [e for e in records if e["kind"] == "prefill"]
        assert len(prefill) == 1 and prefill[0]["round"] == 4 and prefill[0]["tokens"] == 512
        assert all(g["prefill_events"] for g in local if g["from_round"] == 4)
        assert min(e["round"] for e in records if e["kind"] == "decode" and "r4" in e["requests"]) == 5
    return dict(insert=insert, start_state=state, start_s=start, requests=rows, events=records,
                gap_analysis=analysis, local_gaps=local, recovered=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("dynamic", "insert"), default="dynamic")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--no-record", action="store_true", help="关闭阶段记录与NVTX，保留原观测驱动")
    parser.add_argument("--capture", action="store_true", help="仅目标动态运行调用cudaProfilerStart/Stop")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not 0 < args.seconds <= 8 or args.output.resolve().is_relative_to(root):
        parser.error("窗口须为0至8秒，JSON须写在仓库外")
    if args.mode == "insert" and (args.capture or args.no_record) or args.capture and args.no_record:
        parser.error("插入模式保留阶段记录且不采集；--capture须保留记录")
    import torch
    from ampere_kv.bench_external import load_inputs
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION, load_model_and_tokenizer

    prompts, file_hash, ids_hash = load_inputs(args.input, 4, 512, False)
    model, _ = load_model_and_tokenizer()
    templates = [torch.tensor([ids], dtype=torch.long, device="cuda") for ids in prompts]
    with torch.inference_mode():
        scheduler = diagnostic_type()(model, 280, max_batch=4)
        scheduler.nvtx = torch.cuda.nvtx.range
        scheduler.reset_recording(False)
        scheduler.enable_graphs(544, batches=(1, 2, 3, 4))
        warmup = calibration(scheduler, templates, 4, torch.cuda.synchronize)
        result = dict(run_id=args.run_id,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=version("transformers")),
            model=dict(id=MODEL_ID, revision=MODEL_REVISION), input_file_sha256=file_hash, input_ids_sha256=ids_hash,
            config=dict(mode=args.mode, seconds=args.seconds, rate=RATE, input_tokens=512, output_tokens=32,
                graph_batches=sorted(scheduler.graphs), blocks_per_layer=280, max_batch=4, graph_capacity_tokens=544,
                weights_dtype="bf16", kv_dtype="bf16", decode_kernel="v1", greedy=True, ignore_eos=True,
                prefill="whole_sdpa", mixed=False, pipeline=False, structured=False,
                recording=not args.no_record, profiler_control=args.capture,
                time_basis="CPU perf_counter相对每次Observation起点；输出序号从0起，轮次从1起",
                stages="Decode含选词提交；父子CPU区间不相加；GPU关系以共享NVTX标记关联，不相减时钟原点"),
            warmup=warmup, runs=[])
        try:
            if args.mode == "dynamic":
                planned = arrival_table(RATE, args.seconds)
                scheduler.reset_recording(not args.no_record)
                with capture_range(torch, args.capture):
                    sample = run_arrivals(scheduler, templates, planned, args.seconds, 30, torch.cuda.synchronize)
                analysis = gap_analysis(sample["requests"], scheduler.records) if not args.no_record else None
                result["runs"].append(dict(arrivals=planned, arrival_table_sha256=digest(planned),
                                          sample=sample, events=scheduler.records, gap_analysis=analysis))
                print(f"[动态] 完成={sample['counts']} Graph={sample['graph']} 回收={sample['recovered']}")
                if analysis:
                    print(f"[间隔] 含Prefill={analysis['with_prefill']} 无Prefill={analysis['without_prefill']}")
                    for gap in analysis["examples"]:
                        print(f"[案例] {gap}")
            else:
                common = None
                for pair, order in enumerate(((False, True), (True, False)), 1):
                    for insert in order:
                        torch.cuda.synchronize()
                        run = insertion(scheduler, templates, [digest(ids) for ids in prompts], insert, common)
                        if common is None:
                            common = run["start_state"]
                        run["pair"] = pair
                        result["runs"].append(run)
                        print(f"[插入] 第{pair}对 {'B' if insert else 'A'} 起点相同，回收={run['recovered']}")
                        for round_id in range(2, 7):
                            gaps = [g["duration_ms"] for g in run["local_gaps"] if g["from_round"] == round_id]
                            print(f"[局部] round={round_id}→{round_id+1} 旧请求间隔ms={gaps}")
        finally:
            scheduler.reset_recording(False)
            torch.cuda.synchronize()
            if not recovered(scheduler):
                scheduler.observation.cutting = True
                for request in list(scheduler.observation.live.values()):
                    scheduler.cancel(request)
                while scheduler.observation.live:
                    scheduler.step()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] {args.run_id}，JSON={args.output}；CPU阶段关联已保存")


if __name__ == "__main__":
    main()
