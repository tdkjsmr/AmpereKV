"""调度验收：块回收、请求隔离、批量算子与独立生成对照。"""

import torch

from ampere_kv.paged_cache import PagedKVStorage
from ampere_kv.runner import PAGED_BLOCK_SIZE, encode_prompt, generate_tokens, load_model_and_tokenizer
from ampere_kv.scheduler import FINISHED, Scheduler


def _budgets(prompts: list[torch.Tensor], limits: tuple[int, ...]) -> list[int]:
    """按 Scheduler.submit 的同一公式算出每个请求的每层块预算，用例容量由此推得。"""
    return [(p.shape[1] + n + PAGED_BLOCK_SIZE) // PAGED_BLOCK_SIZE for p, n in zip(prompts, limits)]


def _drive(model, prompts, limits, capacity: int, batch: int):
    """提交并跑完一批请求，返回请求列表、调度器、各请求占过的块编号、等待峰值与真实最大成块数。

    完成的请求由 `step()` 交回来，所以这里由调用方自己保存请求引用，调度器不留历史。
    真实最大成块数取每轮 `step()` 报告的批量大小，不从轮末活动队列长度反推。
    """
    scheduler = Scheduler(model, capacity, max_batch=batch)
    requests = [scheduler.submit(f"r{index}", prompt, limit)
                for index, (prompt, limit) in enumerate(zip(prompts, limits))]
    held: dict[str, set[int]] = {}
    peak_waiting = peak_group = 0
    for _ in range(512):
        if all(request.status == FINISHED for request in requests):
            break
        _, sizes = scheduler.step()
        peak_waiting = max(peak_waiting, len(scheduler.waiting))
        peak_group = max(peak_group, max(sizes, default=0))
        for request in scheduler.running:
            held[request.request_id] = set(request.caches[0]._table.block_ids)
    else:
        raise RuntimeError("超过最大调度轮次，请求未全部完成；检查预留额度与释放路径")
    assert scheduler.reserved_blocks == 0, f"收尾预留额度不为零：{scheduler.reserved_blocks}"
    assert all(storage._pool.num_free_blocks == capacity for storage in scheduler.storages), "有请求的块未归还"
    return requests, scheduler, held, peak_waiting, peak_group


def check_lifecycle(model, prompts, limits: tuple[int, ...]) -> None:
    """检查交错与独立执行一致、提前完成隔离、块复用和容量回收。"""
    budgets = _budgets(prompts, limits)
    # 容量取"r1 加上 r0 与 r2 中较大的预算"：r0 与 r1 都在跑时剩余空间必然小于 r2 的预算，
    # r2 一定要等；r0 一释放又必然不小于 r2。用例覆盖因此不依赖具体 Prompt 长度。
    capacity = max(budgets[0], budgets[2]) + budgets[1]
    print(f"[观测] 输入 Token={tuple(p.shape[1] for p in prompts)}，每层预算={budgets}，每层容量={capacity} 块")
    requests, scheduler, held, peak_waiting, _ = _drive(model, prompts, limits, capacity, 1)
    for request, prompt, limit in zip(requests, prompts, limits):
        # 同一 BF16/V1 路径分别单独运行，作为交错调度的对照：Token 序列必须完全一致。
        alone = generate_tokens(model, prompt, max_new_tokens=limit, cache_kind="paged", cuda_decode=True)
        assert request.output_ids == alone, f"{request.request_id} 交错与独立不一致 {request.output_ids} != {alone}"
    assert peak_waiting > 0, "第三个请求从未等待，复用与容量用例没有覆盖"
    print(f"[PASS] 交错执行与独立执行序列一致：三个请求分别生成 {limits} 个 Token")
    print(f"[PASS] 提前完成互不影响：r0 停止原因={requests[0].stop_reason}，r1 仍完整生成 {limits[1]} 个")
    # 只要求有交集：块栈后进先出，r2 接入时先弹到 r0 刚归还的编号，但 r1 之后的增长也可能拿走
    # 其中某个块，所以"子集"太严格。证明没读到别人旧历史的是上面与独立运行一致。
    assert held["r2"] & held["r0"], f"r2 没复用到 r0 归还的块：{sorted(held['r2'])} ∩ {sorted(held['r0'])} 为空"
    print(f"[PASS] 新请求复用归还的块：r2 用到 r0 的块编号 {sorted(held['r2'] & held['r0'])}")
    oversized = prompts[0].new_ones(1, capacity * PAGED_BLOCK_SIZE + 1)
    try:
        scheduler.submit("oversized", oversized, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("预算超过整层容量的请求未被拒绝")
    print(f"[PASS] 容量与回收正确：每层 {capacity} 块，收尾预留额度归零、{len(scheduler.storages)} 层块全部归还")


def check_batched(model, prompts, limits: tuple[int, ...], batch: int) -> None:
    """检查批量前向、逐请求前向与单独运行三方给出同样的 Token 序列。

    用例把同一 Prompt 重复拉长，使各请求历史长度不同：跨过 16 Token 的块边界，最长的一条跨
    过分段长度 64 的多个分段；队首请求先退出让批量缩小，队尾请求在中途接入正在跑的批量。
    """
    capacity = sum(_budgets(prompts, limits)[:batch])
    single, _, _, _, single_group = _drive(model, prompts, limits, capacity, 1)
    batched, _, _, peak_waiting, peak_group = _drive(model, prompts, limits, capacity, batch)
    assert single_group == 1 and peak_group == batch, f"真实成块数 {single_group}/{peak_group} 未覆盖 1 与 {batch}"
    assert peak_waiting > 0, "没有请求等待过，中途接入正在跑的批量这一分支没被覆盖"
    for index, prompt in enumerate(prompts):
        alone = generate_tokens(model, prompt, max_new_tokens=limits[index], cache_kind="paged", cuda_decode=True)
        want_single = single[index].output_ids
        want_batch = batched[index].output_ids
        assert alone == want_single == want_batch, (
            f"r{index} 三方不一致：单独={alone}，逐请求={want_single}，批量={want_batch}")
    print(f"[PASS] 批量 Decode 与逐请求、单独运行三方一致：{len(prompts)} 个请求，历史长度 "
          f"{tuple(p.shape[1] for p in prompts)}，生成上限 {limits}，最大成块 {batch}")


def check_batched_operator(starts: tuple[int, ...] = (0, 16, 64)) -> None:
    """批量接口的算子级对照：一次启动放写入后长度为 1、17、65 的三个请求，对齐连续 KV 参考。

    不加载模型，只验新内核的数值与寻址：定宽块表、跨块与跨 64 分段的行、短请求的空分段，
    以及"本步 K/V 是否落在每个请求自己的起点槽位"。块表填充列一律放 -1，内核只要读到填充列
    就会在设备端断言上失败，所以通过本身就证明填充没被读。
    """
    from ampere_kv import _C
    kv_heads, q_heads, block, dim = 2, 4, PAGED_BLOCK_SIZE, 128
    lengths = [start + 1 for start in starts]
    sizes = [(length + block - 1) // block for length in lengths]
    tables, cursor = [], 0
    for size in sizes:
        tables.append(list(range(cursor, cursor + size)))
        cursor += size
    width = max(sizes)
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    storage = PagedKVStorage(kv_heads, dim, cursor, block, device=device)
    # NaN 哨兵：只有本请求写过的槽位允许变成数字，用它同时证明起点正确和没有多写一格。
    storage._key.fill_(float("nan"))
    storage._value.fill_(float("nan"))
    keys = [torch.randn(1, kv_heads, length, dim, generator=generator, device=device).to(torch.bfloat16)
            for length in lengths]
    values = [torch.randn(1, kv_heads, length, dim, generator=generator, device=device).to(torch.bfloat16)
              for length in lengths]
    for row, key, value, start in zip(tables, keys, values, starts):
        if start:  # 历史用已验证的单请求写入内核放好，批量内核只负责本步那一个 Token。
            _C.bf16_write(key[:, :, :start].contiguous(), value[:, :, :start].contiguous(),
                          storage._key, storage._value, torch.tensor(row, dtype=torch.long, device=device), 0)
    table = torch.tensor([row + [-1] * (width - len(row)) for row in tables], dtype=torch.long, device=device)
    table_starts = torch.tensor(starts, dtype=torch.long, device=device)
    # 每个切片是 [1, KV头, 1, 128]，要按 dim=0 拼成 [批, KV头, 1, 128]；用 stack 会多出一维，
    # 被批量写入入口的"必须是四维"边界检查拒绝。cat 的结果本身就是连续张量。
    new_key = torch.cat([key[:, :, start:] for key, start in zip(keys, starts)], dim=0)
    new_value = torch.cat([value[:, :, start:] for value, start in zip(values, starts)], dim=0)
    _C.bf16_write_batched(new_key, new_value, storage._key, storage._value, table, table_starts)
    queries = torch.randn(len(starts), q_heads, 1, dim, generator=generator, device=device).to(torch.bfloat16)
    got = _C.paged_decode_batched(queries.contiguous(), storage._key, storage._value, table, table_starts)
    group = q_heads // kv_heads
    for index, (key, value, start) in enumerate(zip(keys, values, starts)):
        # 参考是同一条连续 K/V 上的 FP32 Attention：与内核同一份数据，只换寻址方式。
        want = torch.nn.functional.scaled_dot_product_attention(
            queries[index:index + 1].float().contiguous(), key.float().repeat_interleave(group, dim=1).contiguous(),
            value.float().repeat_interleave(group, dim=1).contiguous(), dropout_p=0.0, is_causal=False,
            scale=dim ** -0.5)
        torch.testing.assert_close(got[index:index + 1].float(), want, rtol=0.01, atol=0.002)
        tail = lengths[index]
        row = tables[index]
        torch.testing.assert_close(storage._key[row[start // block], :, start % block],
                                   key[0, :, start], rtol=0, atol=0)
        assert torch.isnan(storage._key[row[tail // block], :, tail % block]).all(), f"r{index} 多写了一格"
    print(f"[PASS] 批量算子对照：同组长度 {tuple(lengths)} 的数值、新写入位置与未写哨兵全部符合；"
          "块表填充列用 -1，若被误读则断言失败")


def main() -> None:
    """先检查批量算子，再检查请求生命周期和批量生成序列。"""
    check_batched_operator()
    model, tokenizer = load_model_and_tokenizer()
    prompts = [encode_prompt(tokenizer, text) for text in
               ("用一句话解释 KV 缓存。", "列举三种可再生能源。", "为什么天空是蓝色的？")]
    check_lifecycle(model, prompts, (4, 8, 2))
    # 重复拉长短句制造不同的历史长度：base 约 20 Token，×6 后约 120 Token（跨多个分段）。
    base = prompts[0]
    check_batched(model, [base, base.repeat(1, 3), base.repeat(1, 6), prompts[1].repeat(1, 2), prompts[2]],
                  (3, 6, 4, 8, 2), 4)
    print("[完成] 整段 Prefill 与批量 Decode 调度检查通过")


if __name__ == "__main__":
    main()
