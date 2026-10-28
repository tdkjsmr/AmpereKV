"""真实模型的固定批量 Graph 调度检查；不计时，不替代独立算子对照。"""

import time

import torch

from ampere_kv.graph_decode import CapturedDecode
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (PAGED_BLOCK_SIZE, load_model_and_tokenizer, model_forward,
                              model_forward_batched, prepare_batched_decode)
from ampere_kv.scheduler import FINISHED, Scheduler


def make_state(model, lengths, steps):
    if not 1 <= len(lengths) <= 4 or min(lengths) < 1 or steps < 1:
        raise ValueError("Graph 对照用例仅支持 B=1至4、非空历史和至少一步 Decode")
    width = max((length + steps + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE for length in lengths)
    blocks = sum((length + steps + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE for length in lengths)
    storages = [PagedKVStorage(model.config.num_key_value_heads, model.config.head_dim,
                               blocks, PAGED_BLOCK_SIZE, device="cuda") for _ in model.model.layers]
    caches = [[PagedKVCache(storage) for _ in lengths] for storage in storages]
    return storages, caches, width


def prefill(model, caches, lengths):
    for request, length in enumerate(lengths):
        ids = torch.arange(1, length + 1, device="cuda", dtype=torch.long).remainder(model.config.vocab_size)[None]
        positions = torch.arange(length, device="cuda", dtype=torch.long)[None]
        model_forward(model, ids, positions, [layer[request] for layer in caches],
                      is_prefill=True, output_logits=False)


def save_prefix(caches):
    return [[cache.get() for cache in layer] for layer in caches]


def restore(storages, caches, prefix):
    for layer in caches:
        for cache in layer:
            cache.release()
    refreshed = [[PagedKVCache(storage) for _ in prefix[0]] for storage in storages]
    for layer, saved in zip(refreshed, prefix):
        for cache, (key, value) in zip(layer, saved):
            cache.append(key, value, fused=True)
    return refreshed


def prepare_layers(caches, width):
    return [prepare_batched_decode(layer, "cuda", width) for layer in caches]


def inputs(model, lengths, step):
    tokens = torch.tensor([[(11 + 37 * request + step * 13) % model.config.vocab_size]
                           for request in range(len(lengths))], dtype=torch.long, device="cuda")
    positions = torch.tensor([[length + step] for length in lengths], dtype=torch.long, device="cuda")
    return tokens, positions


def check_equal(actual, expected, label):
    if not torch.equal(actual, expected):
        first = (actual != expected).nonzero()[0].tolist()
        error = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label} 首个不同索引={first}，最大绝对误差={error:.9g}")


def check_state(paths, addresses, lengths, step):
    reference = paths[0][1]
    for name, caches in paths:
        for layer_index, (layer, ref_layer) in enumerate(zip(caches, reference)):
            ownership = [set(cache._table.block_ids) for cache in layer]
            if any(a & b for index, a in enumerate(ownership) for b in ownership[index + 1:]):
                raise AssertionError(f"{name} 第{layer_index}层出现跨请求块重叠")
            for request, (cache, ref) in enumerate(zip(layer, ref_layer)):
                if cache.length != lengths[request] + step + 1:
                    raise AssertionError(f"{name} 第{layer_index}层请求{request}长度错误")
                if (cache._key.data_ptr(), cache._value.data_ptr()) != addresses[name][layer_index]:
                    raise AssertionError(f"{name} 第{layer_index}层存储地址改变")
                for actual, expected in zip(cache.get(), ref.get()):
                    check_equal(actual, expected, f"{name} 第{layer_index}层请求{request} KV")


def create_paths(model, lengths, steps):
    eager_storage, eager, width = make_state(model, lengths, steps)
    prefill(model, eager, lengths)
    prefix = save_prefix(eager)
    paths = [("旧eager", eager_storage, eager)]
    for name in ("预备eager", "Graph"):
        storage, caches, _ = make_state(model, lengths, steps)
        if name == "Graph":
            # 仅改变第1层空闲栈顺序，让各层的逻辑块落到不同物理编号。
            pool = storage[1]._pool
            held = (pool.allocate(), pool.allocate())
            for block in held:
                pool.free(block)
        paths.append((name, storage, restore(storage, caches, prefix)))
    return paths, prefix, width


def run_step(model, name, caches, width, tokens, positions, captured):
    if name == "旧eager":
        return model_forward_batched(model, tokens, positions, caches)
    if name == "预备eager":
        return model_forward_batched(model, tokens, positions, caches, prepare_layers(caches, width))
    return captured.step(caches, tokens, positions)


def close_paths(paths, blocks):
    for name, storages, caches in paths:
        for layer, storage in zip(caches, storages):
            for cache in layer:
                cache.release()
            if storage._pool.num_free_blocks != blocks:
                raise AssertionError(f"{name} 块未全部归还")


def check_case(model, lengths):
    steps = 4
    paths, prefix, width = create_paths(model, lengths, steps)
    if paths[2][2][0][0]._table.block_ids == paths[2][2][1][0]._table.block_ids:
        raise AssertionError("Graph 用例未覆盖不同层的物理块编号")
    tokens, positions = inputs(model, lengths, 0)
    graph = CapturedDecode(model, paths[2][2], width, tokens, positions)
    captured_table = tuple(tuple(cache._table.block_ids) for cache in paths[2][2][0])
    paths[2] = (paths[2][0], paths[2][1], restore(paths[2][1], paths[2][2], prefix))
    addresses = {name: [(layer[0]._key.data_ptr(), layer[0]._value.data_ptr()) for layer in caches]
                 for name, _, caches in paths}
    seen_tables = set()
    for step in range(steps):
        tokens, positions = inputs(model, lengths, step)
        outputs = [(name, run_step(model, name, caches, width, tokens, positions, graph).clone())
                   for name, _, caches in paths]
        for name, output in outputs[1:]:
            check_equal(output, outputs[0][1], f"B={len(lengths)} 第{step + 1}步 {name} logits")
        for layer, (starts, table) in enumerate(graph.metadata):
            actual = paths[2][2][layer]
            if starts.tolist() != [cache.length - 1 for cache in actual]:
                raise AssertionError(f"Graph 第{layer}层起点映射错误")
            rows = table.tolist()
            for row, cache in zip(rows, actual):
                if row[:len(cache._table.block_ids)] != list(cache._table.block_ids):
                    raise AssertionError(f"Graph 第{layer}层块表映射错误")
        check_state([(name, caches) for name, _, caches in paths], addresses, lengths, step)
        seen_tables.add(tuple(tuple(cache._table.block_ids) for cache in paths[2][2][0]))
    if len(seen_tables) < 2 or all(table == captured_table for table in seen_tables):
        raise AssertionError("Graph 重放未覆盖变化的块表")
    if graph.replays != steps:
        raise AssertionError("Graph 重放次数与真实 Decode 步数不一致")
    close_paths(paths, sum((length + steps + 15) // 16 for length in lengths))
    print(f"[PASS] B={len(lengths)} 历史={tuple(lengths)}：常规eager/静态缓冲eager/Graph 四步 logits、选词、36层KV、地址、块归属与回收严格一致；重放={graph.replays}，块表发生变化")


def prompt(model, length, offset):
    return torch.arange(offset + 1, offset + length + 1, dtype=torch.long,
                        device="cuda").remainder(model.config.vocab_size)[None]


def start_pair(model, lengths, limits, max_batch, max_tokens, *, batches=None):
    blocks = sum((length + limit + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE
                 for length, limit in zip(lengths, limits))
    eager = Scheduler(model, blocks, max_batch=max_batch)
    graph = Scheduler(model, blocks, max_batch=max_batch)
    graph.enable_graphs(max_tokens, batches=batches)
    for field in ("tokens", "positions", "metadata_buffer", "logits"):
        assert len({getattr(g, field).data_ptr() for g in graph.graphs.values()}) == len(graph.graphs)
    assert graph.reserved_blocks == 0 and all(s._pool.num_free_blocks == blocks for s in graph.storages)
    assert eager.graphs is None and eager.graph_calls == eager.graph_fallbacks == eager.graph_captures == 0
    inputs = [prompt(model, length, index * 97) for index, length in enumerate(lengths)]
    references = [eager.submit(f"r{index}", ids, limit, ignore_eos=True)
                  for index, (ids, limit) in enumerate(zip(inputs, limits))]
    requests = [graph.submit(f"r{index}", ids, limit, ignore_eos=True)
                for index, (ids, limit) in enumerate(zip(inputs, limits))]
    addresses = {name: [(s._key.data_ptr(), s._value.data_ptr()) for s in scheduler.storages]
                 for name, scheduler in (("eager", eager), ("graph", graph))}
    return eager, graph, references, requests, addresses, blocks


def compare(eager, graph, references, requests, addresses):
    for reference, request in zip(references, requests):
        assert (request.output_ids, request.status, request.stop_reason) == (
            reference.output_ids, reference.status, reference.stop_reason), (
                request.request_id, request.output_ids, reference.output_ids,
                request.status, reference.status, request.stop_reason, reference.stop_reason)
        if request.caches:
            for layer, (cache, other) in enumerate(zip(request.caches, reference.caches)):
                assert cache.length == other.length, (request.request_id, layer, cache.length, other.length)
    for name, scheduler in (("eager", eager), ("graph", graph)):
        active = [request for request in requests if request.caches] if name == "graph" else [
            request for request in references if request.caches]
        for layer, storage in enumerate(scheduler.storages):
            owned = [set(request.caches[layer]._table.block_ids) for request in active]
            assert all(not first & second for index, first in enumerate(owned) for second in owned[index + 1:])
            for request in active:
                cache = request.caches[layer]
                assert cache.length == request.caches[0].length
                assert (cache._key.data_ptr(), cache._value.data_ptr()) == addresses[name][layer]


def check_uploaded(graph, before, groups, addresses):
    checks = 0
    for batch, captured in graph.graphs.items():
        base, views = addresses[batch]
        assert captured.metadata_buffer.data_ptr() == base
        assert captured.metadata_buffer.numel() == len(captured.metadata) * batch * (captured.width + 1)
        item_size = captured.metadata_buffer.element_size()
        for layer, (starts, table) in enumerate(captured.metadata):
            assert starts.is_contiguous() and table.is_contiguous()
            assert (starts.data_ptr(), table.data_ptr()) == views[layer]
            assert starts.data_ptr() == base + layer * batch * item_size
            assert table.data_ptr() == base + (len(captured.metadata) * batch + layer * batch * captured.width) * item_size
        if captured.replays == before[batch]:
            continue
        # 用例每轮至多四个活动请求；只核对本轮真实命中的那个子批。
        group = next(group for group in groups if len(group) == batch)
        for layer, (device_starts, device_table) in enumerate(captured.metadata):
            starts, rows = device_starts.tolist(), device_table.tolist()
            for index, request in enumerate(group):
                if not request.caches:
                    continue
                cache = request.caches[layer]
                assert starts[index] == cache.length - 1
                assert rows[index][:len(cache._table.block_ids)] == list(cache._table.block_ids)
                checks += 1
    return checks


def drive(model, lengths, limits, max_batch, max_tokens, *, lifecycle=False, batches=None, boundary_check=False):
    eager, graph, references, requests, addresses, blocks = start_pair(
        model, lengths, limits, max_batch, max_tokens, batches=batches)
    captured_batches = set(graph.graphs)
    assert captured_batches == set(batches if batches is not None else ((1, 4) if max_batch >= 4 else (1,)))
    metadata_addresses = {batch: (captured.metadata_buffer.data_ptr(),
                                  [(starts.data_ptr(), table.data_ptr()) for starts, table in captured.metadata])
                          for batch, captured in graph.graphs.items()}
    sizes_seen, boundary, uploads = set(), [], 0
    stopped = added = reused = new_hit = False
    freed = set()
    after_new = None
    post_stop_batches, new_batches = set(), set()
    for _ in range(64):
        if all(request.status == FINISHED for request in requests):
            break
        before = {batch: captured.replays for batch, captured in graph.graphs.items()}
        before_calls = graph.graph_calls
        before_fallbacks = graph.graph_fallbacks
        # 停止发生在轮首，块表核对须使用去掉取消/到期请求后的真实Decode组。
        now = time.monotonic()
        running = [r for r in graph.running if not r.stop_reason and (r.deadline is None or r.deadline > now)]
        groups = [running[offset:offset + graph.max_batch]
                  for offset in range(0, len(running), graph.max_batch)]
        hits = [group for group in groups if len(group) in graph.graphs
                and all(r.caches[0].length < max_tokens for r in group)]
        boundary.extend((max(r.caches[0].length for r in group), "Graph" if group in hits else "eager")
                        for group in groups)
        _, graph_sizes = graph.step()
        _, eager_sizes = eager.step()
        assert graph_sizes == eager_sizes
        assert graph.graph_calls - before_calls == len(hits)
        assert graph.graph_fallbacks - before_fallbacks == len(groups) - len(hits)
        assert {b: g.replays - before[b] for b, g in graph.graphs.items()} == {
            b: sum(len(group) == b for group in hits) for b in captured_batches}
        sizes_seen.update(graph_sizes)
        if stopped:
            post_stop_batches.update(graph_sizes)
        compare(eager, graph, references, requests, addresses)
        uploads += check_uploaded(graph, before, groups, metadata_addresses)
        if added and requests[-1] in running and graph.graph_calls > before_calls:
            new_hit = True
            new_batches.update(len(group) for group in hits if requests[-1] in group)
        if lifecycle and not stopped and graph.graphs[4].replays:
            freed = set(requests[0].caches[0]._table.block_ids) | set(requests[1].caches[0]._table.block_ids)
            eager.cancel(references[0])
            graph.cancel(requests[0])
            deadline = time.monotonic() - 1
            references[1].deadline = requests[1].deadline = deadline
            stopped = True
        elif lifecycle and stopped and not added and requests[0].status == requests[1].status == FINISHED:
            ids = prompt(model, 17, 701)
            references.append(eager.submit("new", ids, 8, ignore_eos=True))
            requests.append(graph.submit("new", ids, 8, ignore_eos=True))
            after_new = graph.graph_calls
            added = True
        if added and requests[-1].caches:
            reused |= bool(freed & set(requests[-1].caches[0]._table.block_ids))
    else:
        raise AssertionError("请求未在64轮内完成")
    assert uploads > 0, "Graph 真实请求的起点与块表未核对"
    assert graph.reserved_blocks == eager.reserved_blocks == 0
    assert all(s._pool.num_free_blocks == blocks for s in graph.storages + eager.storages)
    assert eager.graph_calls == eager.graph_fallbacks == eager.graph_captures == 0
    if lifecycle:
        assert requests[0].stop_reason == "已取消" and requests[1].stop_reason == "已超时"
        assert added and reused and new_hit and graph.graph_calls > after_new and graph.graph_captures == len(captured_batches)
        if {2, 3} <= captured_batches:
            assert {2, 3} <= post_stop_batches and 3 in new_batches
        print(f"[PASS] Graph后取消/到期：后续子批={sorted(post_stop_batches)}，新请求复用块并命中={sorted(new_batches)}；"
              f"邻座序列与eager一致，捕获仍为{graph.graph_captures}")
    elif boundary_check:
        assert (max_tokens - 1, "Graph") in boundary and (max_tokens, "eager") in boundary
        assert graph.graph_calls > 0 and graph.graph_fallbacks > 0 and graph.graph_captures == len(captured_batches)
        print(f"[PASS] B{max_batch}容量边界：最后命中起点={max_tokens - 1}，下一步回退起点={max_tokens}；"
              f"命中={graph.graph_calls}，回退={graph.graph_fallbacks}，捕获={graph.graph_captures}")
    else:
        assert {1, 2, 3, 4} <= sizes_seen
        assert all(g.replays > 0 for g in graph.graphs.values())
        assert (graph.graph_fallbacks == 0) == (captured_batches == {1, 2, 3, 4})
        assert graph.graph_captures == len(captured_batches)
        replays = {b: g.replays for b, g in graph.graphs.items()}
        print(f"[PASS] B4逐步准入/退场：子批={sorted(sizes_seen)}，重放={replays}；"
              f"命中={graph.graph_calls}，回退={graph.graph_fallbacks}，捕获={graph.graph_captures}")
    print(f"[PASS] 逐请求完整输出、停止原因、36层长度/地址/块隔离与归还一致；块表上传核对={uploads}")


def main():
    model, _ = load_model_and_tokenizer()
    late = Scheduler(model, 2, max_batch=1)
    late.submit("late", prompt(model, 1, 0), 1)
    mixed = Scheduler(model, 4, max_batch=4, mixed=True)
    for scheduler in (late, mixed):
        try:
            scheduler.enable_graphs(16)
        except ValueError:
            pass
        else:
            raise AssertionError("首次提交后或混合模式错误启用了 Graph")
    print("[PASS] 首次提交后与 mixed 模式拒绝启用 Graph；默认路径未捕获")
    for lengths in ([15], [15, 16], [15, 16, 63], [15, 16, 63, 64]):
        check_case(model, lengths)
    for batch in (1, 2, 3):
        drive(model, [16] * batch, [20] * batch, batch, 32,
              batches=tuple(range(1, batch + 1)), boundary_check=True)
    drive(model, [17] * 4, [8] * 4, 4, 64, batches=(1, 4))
    drive(model, [17] * 4, [8] * 4, 4, 64, lifecycle=True)
    drive(model, [17] * 4, [8] * 4, 4, 64, batches=(1, 2, 3, 4))
    drive(model, [17] * 4, [8] * 4, 4, 64, lifecycle=True, batches=(1, 2, 3, 4))
    print("[完成] B1至B4 Graph对照、真实批量覆盖、退出/取消/到期、容量回退与块复用通过")


if __name__ == "__main__":
    main()
