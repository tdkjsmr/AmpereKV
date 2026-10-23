"""借鉴ZOS依赖拆分的固定身份单步Decode流水；不是完整DeepInfer实现。"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import time
from types import SimpleNamespace

import torch


@dataclass
class Flight:
    group: tuple
    targets: tuple
    slot: object
    sequence: int


class DecodePipeline:
    """B/C共用设备接续与双槽，仅改变提交时机；advance返回实际发射批量。

    接口返回时最多一轮待提交；advance内部可先发射current，再提交previous，
    临时最多两轮，故“前瞻一轮”不等于任意时刻只有一份未提交输出。
    cache.length为已登记执行长度；CPU已提交长度是P+len(output_ids)-1。
    只通过advance/drain驱动；submit/cancel仍沿用Scheduler，禁止外部改running。
    """

    def __init__(self, scheduler, *, overlap):
        self.scheduler, self.overlap = scheduler, overlap
        self.compute, self.copy = torch.cuda.Stream(), torch.cuda.Stream()
        self.slots = {}
        for batch, graph in scheduler.graphs.items():
            size = graph.metadata_buffer.numel() + 2 * batch
            self.slots[batch] = [SimpleNamespace(
                host_meta=torch.empty(size, dtype=torch.long, pin_memory=True),
                device_meta=torch.empty(size, dtype=torch.long, device=scheduler.device),
                tokens=torch.empty(batch, 1, dtype=torch.long, device=scheduler.device),
                output=torch.empty(batch, dtype=torch.long, pin_memory=True),
                sampled=torch.cuda.Event(), ready=torch.cuda.Event()) for _ in range(2)]
        self.pending = self.seed = None
        self.serial = 0
        self.stats = Counter()
        self.last_batches = []
        self.used_fallback = False
        self.tracing = False

    def trace(self, label):
        return torch.cuda.nvtx.range(label) if self.tracing else nullcontext()

    def _stopped(self, request, now):
        if request.deadline is not None and request.deadline <= now:
            request.stop_reason = request.stop_reason or "已超时"
        return bool(request.stop_reason)

    def _same(self, flight, group):
        return flight is not None and flight.group == group and all(
            target == len(r.output_ids) for r, target in zip(group, flight.targets))

    def _eligible(self, group):
        s = self.scheduler
        return (not s.waiting and s.prefilling is None and len(group) in s.graphs
                and all(r.caches[0].length < s.graph_max_tokens for r in group))

    @torch.no_grad()
    def _launch(self, group, previous=None):
        s = self.scheduler
        self.serial += 1
        slot = self.slots[len(group)][self.serial % 2]
        targets = tuple(t + 1 for t in previous.targets) if previous else tuple(len(r.output_ids) + 1 for r in group)
        flight = Flight(group, targets, slot, self.serial)
        caches = [[r.caches[layer] for r in group] for layer in range(len(s.storages))]
        initial = [0] * len(group) if previous else [r.output_ids[-1] for r in group]
        with self.trace(f"pipeline_forward:seq={flight.sequence},batch={len(group)}"), torch.cuda.stream(self.compute):
            logits = s.graphs[len(group)].step_device(caches, previous.slot.tokens if previous else None, slot, initial)
            torch.argmax(logits[:, 0], dim=-1, out=slot.tokens[:, 0])
            slot.sampled.record(self.compute)
        with torch.cuda.stream(self.copy):
            self.copy.wait_event(slot.sampled)
            slot.output.copy_(slot.tokens[:, 0], non_blocking=True)
            slot.ready.record(self.copy)
        s.graph_calls += 1
        self.last_batches.append((len(group), True))
        self.stats["forwards"] += 1
        self.stats["device_tokens"] += len(group)
        return flight

    def _post(self, flight, now):
        # 只等待previous的D2H事件，不等待计算流上已经发射的current。
        with self.trace(f"pipeline_wait:seq={flight.sequence}"):
            flight.slot.ready.synchronize()
        with self.trace(f"pipeline_post:seq={flight.sequence}"):
            for r, target, token in zip(flight.group, flight.targets, flight.slot.output.tolist()):
                if self._stopped(r, now):
                    self.stats["discarded_tokens"] += 1
                    continue
                assert target == len(r.output_ids) + 1, "在途结果身份或顺序错误"
                self.scheduler._commit(r, token)
                self.stats["committed_tokens"] += 1

    def drain(self, now=None):
        """提交最后在途结果或丢弃已停止请求的结果，设备安全后才退休。"""
        now = time.monotonic() if now is None else now
        started = time.perf_counter()
        if self.pending is not None:
            self._post(self.pending, now)
            self.pending = None
            self.stats["drains"] += 1
        self.seed = None
        finished = self.scheduler._stop_round(now)
        self.stats["drain_ms"] += (time.perf_counter() - started) * 1000
        return finished

    def advance(self):
        """轮首观察停止；可返回零输出，或在排空时提交至多两轮合法输出。"""
        s = self.scheduler
        now = time.monotonic()
        self.last_batches, self.used_fallback = [], False
        group = tuple(s.running)
        stopped = any(self._stopped(r, now) for r in group)
        if self.pending is not None:
            previous = self.pending
            # 上限已知时不多算；批次变动、取消、准入或容量边界一律先排空。
            ahead = (self.overlap and not stopped and previous.group == group and self._eligible(group)
                     and all(t < r.max_new_tokens for r, t in zip(group, previous.targets)))
            if ahead:
                self.pending = self._launch(group, previous)
                self.stats["lookahead_forwards"] += 1
                self.stats["max_inflight"] = max(self.stats["max_inflight"], 2)
                self._post(previous, now)
                if any(r.stop_reason for r in group):
                    finished = self.drain(now)
                else:
                    finished = []
                return finished, [n for n, _ in self.last_batches]
            return self.drain(now), []
        finished = s._stop_round(now)
        group = tuple(s.running)
        if not self._eligible(group):
            self.seed = None
            self.used_fallback = True
            self.stats["fallback_steps"] += 1
            more, sizes = s.step()
            self.stats["fallback_forwards"] += len(sizes)
            self.stats["fallback_tokens"] += sum(sizes)
            return finished + more, sizes
        # 回退/Prefill在调用流执行；只在重入时给计算流建立依赖，不全卡同步。
        self.compute.wait_stream(torch.cuda.current_stream())
        source = self.seed if self._same(self.seed, group) else None
        self.pending = self._launch(group, source)
        self.stats["max_inflight"] = max(self.stats["max_inflight"], 1)
        if self.overlap:
            return finished, [len(group)]
        current = self.pending
        self._post(current, now)
        self.pending = None
        finished += s._stop_round(now)
        self.seed = current if not finished else None
        return finished, [len(group)]
