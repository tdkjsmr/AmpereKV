"""单请求草稿Graph：固定物理池、设备候选与同流Token接续。"""

import time
from types import SimpleNamespace

import torch

from ampere_kv.graph_decode import capture_scheduler_graph
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage


class DraftGraph:
    def __init__(self, model, capacity):
        self.model = model
        self.capacity = (capacity + 15) // 16 * 16
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        config = model.config
        self.storages = [PagedKVStorage(config.num_key_value_heads, config.head_dim,
                                        self.capacity // 16, 16, device=model.device)
                         for _ in model.model.layers]
        self.caches = [PagedKVCache(storage) for storage in self.storages]
        self.layers = [[cache] for cache in self.caches]
        self.graph = capture_scheduler_graph(model, self.storages, 1, self.capacity)
        self.tokens = torch.empty(1, 5, dtype=torch.long, device=model.device)
        size = self.graph.metadata_buffer.numel() + 2
        # 每轮最多四步及一次补算各占一槽；验证统一取词后才复用，避免覆盖在途H2D源。
        self.slots = [SimpleNamespace(
            metadata_host=torch.empty(size, dtype=torch.long, pin_memory=True),
            metadata_device=torch.empty(size, dtype=torch.long, device=model.device)) for _ in range(5)]
        torch.cuda.synchronize()
        self.initialization = dict(wall_ms=(time.perf_counter() - started) * 1000,
            allocated_delta_bytes=torch.cuda.memory_allocated() - before,
            capacity=self.capacity, layers=len(self.caches), captures=1)
        self.active = False
        self.device_tokens = self.repairs = 0

    def acquire(self, capacity):
        if self.active or capacity > self.capacity:
            raise ValueError("草稿图已被占用或请求超过固定容量")
        self.active = True
        return self.caches

    def step(self, source, slot):
        return self.graph.step_device(self.layers, source, self.slots[slot], [0])

    def propose(self, pending, count):
        self.tokens[:, 0].fill_(pending)
        for index in range(count):
            logits = self.step(self.tokens[:, index:index + 1], index)
            torch.argmax(logits[:, -1], dim=-1, out=self.tokens[:, index + 1])
            self.device_tokens += 1
        # 各候选各占一个位置；下轮改写前，验证和真实补算已在同一流中排在前面。
        return self.tokens[:, :count + 1]

    def repair(self, source):
        self.step(source, 4)
        self.repairs += 1

    def counts(self):
        return dict(replays=self.graph.replays, device_tokens=self.device_tokens,
                    repairs=self.repairs, captures=1)

    def release(self):
        torch.cuda.current_stream().synchronize()
        for cache in self.caches:
            cache.release()
        self.active = False
