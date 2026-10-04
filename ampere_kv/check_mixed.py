"""混合前向与调度对照：B 个 Decode Token 与一个 Prefill 块共用一次逐 Token 计算。

这里只做独立前向与调度接入对照；不测性能。
"""

import argparse

import torch

from ampere_kv.paged_cache import PagedKVCache
from ampere_kv.runner import (encode_prompt, load_model_and_tokenizer, model_forward,
                              model_forward_batched, model_forward_mixed)
from ampere_kv.scheduler import FINISHED, Scheduler


def make_prompts(tokenizer, batch: int, history: int) -> list[torch.Tensor]:
    """生成长度固定、内容不同的合成输入，只用于隔离与数值对照。"""
    prompts = []
    for index in range(batch):
        ids = encode_prompt(tokenizer, f"用一句话解释 KV 缓存的第 {index} 种说法。")
        prompts.append(ids.repeat(1, -(-history // ids.shape[1]))[:, :history])
    return prompts


def positions(ids, start):
    """位置由本请求历史决定，不能使用打包后的全局Token下标。"""
    return torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)


def prepare(model, prompts, pending, history, extra):
    """同一套池中各请求独占块；两条被测路径各自创建一套，不共享已分配块。"""
    capacity = sum((p.shape[1] + extra + 15) // 16 for p in prompts) + (pending.shape[1] + 15) // 16
    owner = Scheduler(model, capacity)
    caches = [[PagedKVCache(storage) for storage in owner.storages] for _ in range(len(prompts) + 1)]
    tokens = []
    for prompt, cache in zip(prompts, caches):
        out = model_forward(model, prompt, positions(prompt, 0), cache, is_prefill=True)
        tokens.append(out[:, 0].argmax(-1))
    if history:
        prefix = pending[:, :history]
        model_forward(model, prefix, positions(prefix, 0), caches[-1], is_prefill=True, output_logits=False)
    return owner, caches, torch.stack(tokens).reshape(-1, 1)


def release(state):
    """检查只放在对照入口；退出时归还所有请求的块。"""
    owner, caches, _ = state
    for request in caches:
        for cache in request:
            cache.release()
    assert all(s._pool.num_free_blocks == owner.blocks_per_layer for s in owner.storages)


def compare(actual, expected, label):
    """跨GEMM形状可能有舍入差异：观测logits，要求本次贪心选择一致，不放宽旧容差。"""
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual.float() - expected.float()
    relative = delta.norm() / expected.float().norm().clamp_min(1e-12)
    print(f"[观测] {label}：最大误差={delta.abs().max().item():.6g}，相对L2={relative.item():.6g}")
    got, want = actual.argmax(-1), expected.argmax(-1)
    assert torch.equal(got, want), f"{label}选词分歧：混合={got.tolist()}，分离={want.tolist()}"


@torch.inference_mode()
def run_case(model, tokenizer, batch, history, chunks):
    """相同初始化后连续追加；旧请求由分离参考选词驱动，两侧始终使用同一输入历史。"""
    prompts = [make_prompts(tokenizer, batch, 17 + 7 * i)[i] for i in range(batch)]
    pending = make_prompts(tokenizer, 1, history + sum(chunks))[0]
    states, outputs = {}, {}
    try:
        for name in ("分离", "混合"):
            states[name] = prepare(model, prompts, pending, history, len(chunks))
        # 初始化的历史应逐位相同，避免把准备差异算进混合计算误差。
        for left, right in zip(states["分离"][1], states["混合"][1]):
            for a, b in zip(left, right):
                if a.length:
                    for x, y in zip(a.get(), b.get()):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
        addresses = {name: [(s._key.data_ptr(), s._value.data_ptr()) for s in state[0].storages]
                     for name, state in states.items()}
        decode_ids = states["分离"][2]
        start = history
        for step, chunk in enumerate(chunks):
            ids = pending[:, start:start + chunk]
            final = step == len(chunks) - 1
            order = ("混合", "分离") if step % 2 else ("分离", "混合")
            for name in order:
                owner, caches, _ = states[name]
                layer_caches = list(map(list, zip(*caches[:-1])))
                pos = torch.tensor([[c[0].length] for c in caches[:-1]], device="cuda")
                ppos = positions(ids, start)
                if name == "混合":
                    out = model_forward_mixed(model, decode_ids, pos, layer_caches,
                                              ids, ppos, caches[-1], output_prefill_logits=final)
                else:
                    decoded = model_forward_batched(model, decode_ids, pos, layer_caches)
                    prefilled = model_forward(model, ids, ppos, caches[-1], is_prefill=True,
                                              output_logits=final)
                    out = decoded, prefilled
                outputs[name] = out
                for i, request in enumerate(caches):
                    length = start + chunk if i == batch else prompts[i].shape[1] + step + 1
                    assert all(cache.length == length for cache in request)
                for layer, storage in enumerate(owner.storages):
                    assert addresses[name][layer] == (storage._key.data_ptr(), storage._value.data_ptr())
                    blocks = [block for request in caches for block in request[layer]._table.block_ids]
                    assert len(blocks) == len(set(blocks)), "请求物理块重叠"
            compare(outputs["混合"][0], outputs["分离"][0], f"B={batch} H={start} C={chunk} Decode")
            if final:
                compare(outputs["混合"][1], outputs["分离"][1], "Prefill末块")
            else:
                assert outputs["混合"][1] is None and outputs["分离"][1] is None
            decode_ids = outputs["分离"][0][:, 0].argmax(-1).unsqueeze(1)
            start += chunk
    finally:
        for state in states.values():
            release(state)
    print(f"[PASS] B={batch} 初始H={history} 块长={chunks}：同历史选词、缓存长度、块隔离、地址与回收通过")


@torch.inference_mode()
def run_schedule(model, tokenizer, batch, chunk, *, mixed, max_batch=None, stop_after_mixed=None,
                 limits=None):
    """真实调度跑一遍：返回每请求序列、实际 Decode 轮数、新请求首 Token 轮次与收尾账目。

    既有请求用不同的生成上限，一次覆盖"中途正常完成"与"空位后新请求准入"两类分支；新请求
    Prompt 为 `3*C+1`，所以混合轮既遇到中间块、也遇到 1 Token 末块。`stop_after_mixed` 在跑满
    若干次新请求参与的混合轮后取消它；其余等待请求仍可正常参与混合。
    """
    prompts = [make_prompts(tokenizer, 1, 33 + 11 * i)[0] for i in range(batch)]
    limits = list(limits) if limits is not None else [4, 6, 3, 5, 2, 2, 2, 2][:batch]
    pending = make_prompts(tokenizer, 1, 3 * chunk + 1)[0]
    capacity = (sum((p.shape[1] + n + 15) // 16 for p, n in zip(prompts, limits))
                + (pending.shape[1] + 8 + 15) // 16)
    scheduler = Scheduler(model, capacity, max_batch=max_batch or batch, prefill_chunk_size=chunk, mixed=mixed)
    requests = [scheduler.submit(f"r{index}", prompt, limit, ignore_eos=True)
                for index, (prompt, limit) in enumerate(zip(prompts, limits))]
    new = scheduler.submit("new", pending, 8, ignore_eos=True)
    tracked = requests + [new]
    decoded = {request.request_id: 0 for request in tracked}
    mixed_flags, first = [], None
    mixed_with_new, cancelled_round = 0, None
    for rounds in range(64):
        active = [request.request_id for request in scheduler.running]  # 本轮起点参与 Decode 的集合
        was_mixed = scheduler.mixed_rounds
        prefilling_new = scheduler.prefilling is new
        admitting_new = scheduler.prefilling is None and bool(scheduler.waiting) and scheduler.waiting[0] is new
        if (stop_after_mixed is not None and mixed_with_new >= stop_after_mixed
                and scheduler.prefilling is new and not new.stop_reason):
            scheduler.cancel(new)  # 只在轮次边界取消，已发出的 GPU 工作不打断
            cancelled_round = rounds
        scheduler.step()
        for request_id in active:
            decoded[request_id] += 1
        mixed_flags.append(scheduler.mixed_rounds > was_mixed)
        if (mixed_flags[-1] and cancelled_round != rounds
                and (prefilling_new or (admitting_new and scheduler.prefilling is new))):
            mixed_with_new += 1
        if first is None and new.output_ids:
            first = rounds
        if all(request.status == FINISHED for request in tracked):
            break
    else:
        raise AssertionError(f"调度未在 64 轮内收尾：已 Decode {decoded}")
    return {"seqs": [request.output_ids for request in requests], "new_seq": new.output_ids,
            "decoded": decoded, "first": first, "limits": limits, "mixed_flags": mixed_flags,
            "mixed_with_new": mixed_with_new, "cancelled_round": cancelled_round,
            "mixed_rounds": scheduler.mixed_rounds, "fallback_rounds": scheduler.fallback_rounds,
            "reserved": scheduler.reserved_blocks, "capacity": capacity, "new_status": new.status,
            "new_reason": new.stop_reason,
            "free": all(storage._pool.num_free_blocks == capacity for storage in scheduler.storages)}


def check_schedule(model, tokenizer, batch=4, chunk=16) -> None:
    """混合与分离调度的同负载对照，外加"混合轮后取消"与"超过一批回退"两个小覆盖。

    序列分歧先报告轮次与序列，不放宽容差也不删检查；`mixed` 与 `separate` 的差异来自打包形状
    改变线性层执行形状，逐位相同从来不是前提，所以比的是每请求选词序列。
    """
    separate = run_schedule(model, tokenizer, batch, chunk, mixed=False)
    mixed = run_schedule(model, tokenizer, batch, chunk, mixed=True)
    assert separate["mixed_rounds"] == 0, f"默认路径混进了混合轮：{separate['mixed_rounds']}"
    assert mixed["mixed_rounds"] > 0 and mixed["fallback_rounds"] == 0, f"混合开关没生效：{mixed}"
    assert mixed["seqs"] == separate["seqs"] and mixed["new_seq"] == separate["new_seq"], (
        f"选词分歧：旧请求混合={mixed['seqs']} 分离={separate['seqs']}；"
        f"新请求混合={mixed['new_seq']} 分离={separate['new_seq']}；先定位轮次与 logits，不放宽检查")
    assert mixed["decoded"] == separate["decoded"], (
        f"实际 Decode 次数不一致：混合={mixed['decoded']}，分离={separate['decoded']}")
    assert mixed["first"] == separate["first"], (
        f"新请求首 Token 轮次不一致：混合={mixed['first']}，分离={separate['first']}")
    for name, result in (("分离", separate), ("混合", mixed)):
        assert result["reserved"] == 0 and result["free"], f"{name}收尾账目不对：{result}"
        assert all(len(seq) == limit for seq, limit in zip(result["seqs"], result["limits"])), (
            f"{name}路径没有请求按各自上限正常完成：{[len(s) for s in result['seqs']]}")
    print(f"[PASS] 调度对照 B={batch} 块长={chunk}：混合 {mixed['mixed_rounds']} 轮、回退 0 轮，"
          f"序列/Decode 次数/新请求首 Token 轮次({mixed['first']})与分离一致，"
          f"额度归零、每层 {mixed['capacity']} 块全归还")
    cancelled = run_schedule(model, tokenizer, batch, chunk, mixed=True, stop_after_mixed=1)
    flags = cancelled["mixed_flags"]
    assert cancelled["new_status"] == FINISHED and cancelled["new_reason"] == "已取消", f"{cancelled}"
    assert cancelled["first"] is None and not cancelled["new_seq"], f"被取消的新请求仍选了词：{cancelled}"
    assert cancelled["mixed_with_new"] == 1 and cancelled["cancelled_round"] is not None, (
        f"新请求没有在一次混合轮后取消：{cancelled}")
    assert all(len(seq) == limit for seq, limit in zip(cancelled["seqs"], cancelled["limits"])), (
        f"取消新请求后邻座没继续到各自上限：{[len(s) for s in cancelled['seqs']]}")
    assert cancelled["reserved"] == 0 and cancelled["free"], f"取消后收尾账目不对：{cancelled}"
    print(f"[PASS] 混合轮后取消新请求：参与 {cancelled['mixed_with_new']} 轮后取消，"
          f"取消轮次={cancelled['cancelled_round']}，总混合轮={sum(flags)}；"
          "新请求未选词即归还块，其余请求仍按各自上限完成")
    # 两条旧请求要活到新请求开始 Prefill，才能实际覆盖活动数超过一批的回退分支。
    fallback = run_schedule(model, tokenizer, 2, chunk, mixed=True, max_batch=1, limits=(12, 14))
    assert fallback["fallback_rounds"] > 0, f"没走到超过一批的回退分支：{fallback}"
    assert all(len(seq) == limit for seq, limit in zip(fallback["seqs"], fallback["limits"])), (
        f"回退轮里请求没跑完：{[len(s) for s in fallback['seqs']]}")
    assert fallback["reserved"] == 0 and fallback["free"], f"回退收尾账目不对：{fallback}"
    print(f"[PASS] 活动数超过一批时回退分离路径：B=2/max_batch=1 下 {fallback['fallback_rounds']} 轮回退"
          f"（同场另有 {fallback['mixed_rounds']} 轮单请求混合），回退轮不算混合成功，"
          f"请求仍各自跑满上限且块全归还")


def main():
    parser = argparse.ArgumentParser(description="BF16混合前向与调度对照")
    parser.parse_args()
    model, tokenizer = load_model_and_tokenizer()
    for batch, history in ((1, 0), (4, 17)):
        run_case(model, tokenizer, batch, history, (16, 16, 1))
    check_schedule(model, tokenizer)
    print("[完成] 独立混合前向 + 可选混合调度接入对照通过；未做 INT8、Graph 与自适应块长，"
          "非自由生成或 SLO 验收")


if __name__ == "__main__":
    main()
