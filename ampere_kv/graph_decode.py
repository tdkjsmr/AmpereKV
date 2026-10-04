"""固定批量 BF16 Decode Graph：捕获、元数据上传与重放。"""

import time

import torch

from ampere_kv.paged_cache import PagedKVCache
from ampere_kv.runner import PAGED_BLOCK_SIZE, model_forward_batched


class CapturedDecode:
    """图长期持有静态输入、连续元数据视图、输出及物理 KV 存储。"""

    def __init__(self, model, caches, width, tokens, positions):
        batch = len(caches[0])
        self.model = model
        self.captured_caches = caches
        self.width = width
        before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        started = time.perf_counter()
        self.tokens = torch.empty_like(tokens)
        self.positions = torch.empty_like(positions)
        layers = len(caches)
        self.metadata_buffer = torch.empty(layers * batch * (width + 1), dtype=torch.long, device=tokens.device)
        starts = self.metadata_buffer[:layers * batch].view(layers, batch)
        tables = self.metadata_buffer[layers * batch:].view(layers, batch, width)
        self.metadata = [(starts[layer], tables[layer]) for layer in range(layers)]
        self.tokens.copy_(tokens)
        self.positions.copy_(positions)
        self.upload_metadata(caches)

        # 侧流预热与捕获只覆盖已经登记的合法位置；宿主长度不能随重放自动前进。
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                model_forward_batched(model, self.tokens, self.positions, caches, self.metadata)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits = model_forward_batched(model, self.tokens, self.positions, caches, self.metadata)
        torch.cuda.synchronize()
        self.init_ms = (time.perf_counter() - started) * 1000
        self.extra_bytes = (torch.cuda.memory_allocated() - before[0],
                            torch.cuda.memory_reserved() - before[1])
        self.replays = 0

    def upload_metadata(self, caches):
        if any(cache.length >= self.width * cache._key.shape[2] for layer in caches for cache in layer):
            raise ValueError("固定块表容量不足")
        starts, tables = [], []
        for layer in caches:
            starts.extend(cache.length for cache in layer)
            for cache in layer:
                row = cache.reserve(1)
                tables.extend(row)
                tables.extend([0] * (self.width - len(row)))
        # 前半段按层存起点，后半段按层、请求存块表；当前流一次阻塞上传。
        host = torch.tensor(starts + tables, dtype=torch.long)
        self.metadata_buffer.copy_(host, non_blocking=False)

    def step(self, caches, tokens, positions):
        self.tokens.copy_(tokens)
        self.positions.copy_(positions)
        self.upload_metadata(caches)
        self.graph.replay()
        self.replays += 1
        # 下一次 replay 会覆盖这个输出；调用方必须当步读取或克隆。
        return self.logits


def capture_scheduler_graph(model, storages, batch, max_tokens):
    """在调度器原有物理池上捕获；临时请求只借块，不成为图的所有者。"""
    caches = [[PagedKVCache(storage) for _ in range(batch)] for storage in storages]
    key = torch.zeros(1, storages[0]._key.shape[1], 1, storages[0]._key.shape[3],
                      dtype=torch.bfloat16, device=storages[0]._key.device)
    for layer in caches:
        for cache in layer:
            cache.append(key, key, fused=True)
    tokens = torch.zeros(batch, 1, dtype=torch.long, device=key.device)
    positions = torch.ones_like(tokens)
    try:
        return CapturedDecode(model, caches, max_tokens // PAGED_BLOCK_SIZE, tokens, positions)
    finally:
        torch.cuda.synchronize()
        for layer in caches:
            for cache in layer:
                cache.release()


if __name__ == "__main__":
    import argparse
    from ampere_kv.check_graph_scheduler import main

    argparse.ArgumentParser(description="Graph 正确性检查；性能入口为 bench_external").parse_args()
    main()
