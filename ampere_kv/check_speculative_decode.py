"""投机解码检查：默认CPU小用例，--model加载固定两模型做同历史与完整生成对照。"""

import argparse

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import encode_prompt
from ampere_kv.speculative_decode import (eos_ids, forward_tokens, generate_speculative,
                                         load_pair, make_caches, select_prefix)


def check_prefix():
    cases = [
        ([4, 5], [4, 5, 6], 8, False, [4, 5, 6], 2, 2),
        ([4, 5], [7, 5, 6], 8, False, [7], 0, 0),
        ([4, 5, 6, 7], [4, 5, 9, 7, 8], 8, False, [4, 5, 9], 2, 2),
        ([4, 0], [4, 0, 6], 8, False, [4, 0], 2, 2),
        ([4, 5], [0, 5, 6], 8, False, [0], 0, 0),
        ([4, 5], [4, 5, 0], 8, False, [4, 5, 0], 2, 2),
        ([4, 0], [4, 0, 6], 8, True, [4, 0, 6], 2, 2),
        ([0, 5], [0, 5, 6], 8, False, [0], 1, 2),
        ([], [7], 1, False, [7], 0, 0),
        ([4], [4, 8], 2, False, [4, 8], 1, 1),
    ]
    for proposals, predictions, remaining, ignore, output, accepted, matched in cases:
        assert select_prefix(proposals, predictions, remaining, {0}, ignore) == (output, accepted, matched)
    print("[PASS] 全接受/首处及中间拒绝、额外与修正Token、EOS裁切和短剩余预算；接受计数不含额外Token")


def check_truncate():
    storage = PagedKVStorage(2, 8, 4, 16)
    cache = PagedKVCache(storage)
    key = torch.arange(560).reshape(1, 2, 35, 8).to(torch.bfloat16)
    value = -key
    addresses = (storage._key.data_ptr(), storage._value.data_ptr())
    try:
        cache.append(key, value)
        for length in (35, 33, 17, 16, 0):
            old_blocks = cache._table.block_ids
            cache.truncate(length)
            assert cache.length == length and len(cache._table.block_ids) == (length + 15) // 16
            assert cache._table.block_ids == old_blocks[:(length + 15) // 16]
            assert storage._pool.num_free_blocks == 4 - (length + 15) // 16
            for actual, source in zip(cache.get(), (key, value)):
                torch.testing.assert_close(actual, source[:, :, :length], rtol=0, atol=0)
            try:
                cache._table.locate(length)
            except ValueError:
                pass
            else:
                raise AssertionError("截断后的无效尾部仍可定位")
            cache.append(key[:, :, length:], value[:, :, length:])
            for actual, source in zip(cache.get(), (key, value)):
                torch.testing.assert_close(actual, source, rtol=0, atol=0)
            assert (storage._key.data_ptr(), storage._value.data_ptr()) == addresses
            assert set(cache._table.block_ids) == set(old_blocks), "追加没有复用归还的尾块"
        for invalid in (-1, 36, 1.5, True):
            before = (cache.length, cache._table.block_ids, storage._pool.num_free_blocks)
            try:
                cache.truncate(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法截断未拒绝：{invalid}")
            assert before == (cache.length, cache._table.block_ids, storage._pool.num_free_blocks)
    finally:
        cache.release()
    assert storage._pool.num_free_blocks == 4
    print("[PASS] 同块/跨块/块边界/零/不变截断，前缀不变、追加覆盖、尾块复用及地址/回收")


def release(caches):
    for cache in caches:
        cache.release()
        assert cache._pool.num_free_blocks == cache._key.shape[0]


def candidates(logits):
    scores, ids = logits.float().topk(2)
    return dict(ids=ids.tolist(), scores=scores.tolist(), gap=(scores[0] - scores[1]).item())


@torch.inference_mode()
def check_draft_hf(draft, ids):
    caches = make_caches(draft, ids.shape[1] + 4)
    reference_cache, current = None, ids
    try:
        for step in range(4):
            start = caches[0].length
            positions = torch.arange(start, start + current.shape[1], device=ids.device)[None]
            actual = forward_tokens(draft, current, caches)
            expected = draft(input_ids=current, position_ids=positions, cache_position=positions[0],
                             past_key_values=reference_cache, use_cache=True, logits_to_keep=1, return_dict=True)
            reference_cache = expected.past_key_values
            torch.testing.assert_close(actual, expected.logits, rtol=0, atol=0)
            assert all(cache.length == reference_cache.get_seq_length(i) for i, cache in enumerate(caches))
            current = ids.new_tensor([[expected.logits[0, -1].argmax().item()]])
    finally:
        release(caches)
    print("[PASS] 0.6B缓存Prefill及三步Decode与其HF logits逐位一致，独立层数/RoPE/绑定权重与回收")


def compare_logits(actual, expected, label):
    difference = actual.float() - expected.float()
    relative = difference.norm() / expected.float().norm().clamp_min(1e-12)
    print(f"[观测] {label}：最大绝对误差={difference.abs().max().item():.7g}，相对L2={relative.item():.7g}")
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    for row in range(actual.shape[1]):
        if actual[0, row].argmax().item() != expected[0, row].argmax().item():
            print(f"[分歧] {label} 行={row}，多位置={candidates(actual[0, row])}，参考={candidates(expected[0, row])}")
            raise AssertionError("同历史选词分歧，保留候选后停止")


@torch.inference_mode()
def check_verification(target, ids):
    for k in (2, 4):
        sets = [make_caches(target, ids.shape[1] + 5) for _ in range(3)]
        current = ids.new_tensor([[11, 13, 17, 19, 23]])[:, :k + 1]
        try:
            for caches in sets:
                forward_tokens(target, ids, caches)
            actual = forward_tokens(target, current, sets[0], all_logits=True)
            last = forward_tokens(target, current, sets[1])
            sequential = torch.cat([forward_tokens(target, current[:, i:i + 1], sets[2])
                                    for i in range(k + 1)], dim=1)
            assert actual.shape == (1, k + 1, target.config.vocab_size)
            assert last.shape == (1, 1, target.config.vocab_size)
            compare_logits(actual[:, -1:], last, f"K{k}新增all_logits末行/原接口")
            compare_logits(actual, sequential, f"K{k}各位置/同历史逐Token")
            assert all(cache.length == ids.shape[1] + k + 1 for caches in sets for cache in caches)
        finally:
            for caches in sets:
                release(caches)
    print("[PASS] 8B K2/K4多位置返回、末位置接口与同历史逐Token选词/位置一致")


@torch.inference_mode()
def ordinary(target, ids, limit, ignore_eos):
    caches = make_caches(target, ids.shape[1] + limit)
    output, tops, current = [], [], ids
    try:
        for _ in range(limit):
            logits = forward_tokens(target, current, caches)
            token = logits[0, -1].argmax().item()
            output.append(token)
            tops.append(candidates(logits[0, -1]))
            if not ignore_eos and token in eos_ids(target):
                break
            current = ids.new_tensor([[token]])
        assert all(cache.length == ids.shape[1] + len(output) - 1 for cache in caches)
    finally:
        release(caches)
    return output, tops


def check_generation(target, draft, ids, limit, ignore_eos):
    reference, tops = ordinary(target, ids, limit, ignore_eos)
    for k, bound in ((2, limit), (4, limit), (4, 1), (4, 2)):
        addresses = {}

        def observe(event, targets, drafts, logits):
            assert len(targets) == len(target.model.layers) and len(drafts) == len(draft.model.layers)
            expected_length = ids.shape[1] + event["output_start"] + len(event["committed"]) - 1
            assert event["length"] == expected_length
            for name, caches in (("target", targets), ("draft", drafts)):
                pointers = tuple((c._key.data_ptr(), c._value.data_ptr()) for c in caches)
                assert pointers == addresses.setdefault(name, pointers)
                assert all(c.length == expected_length and len(c._table.block_ids) == (expected_length + 15) // 16
                           for c in caches)
                assert all(c._pool.num_free_blocks + len(c._table.block_ids) == c._key.shape[0] for c in caches)
            assert not {c._key.data_ptr() for c in targets} & {c._key.data_ptr() for c in drafts}
            for index, token in enumerate(event["committed"]):
                position = event["output_start"] + index
                if position >= len(reference) or token != reference[position]:
                    print(f"[分歧] K={k} 首个输出位置={position}，轮输入={event.get('input_ids')}，"
                          f"目标普通={tops[position] if position < len(tops) else None}，"
                          f"验证={candidates(logits[0, index])}，提交={token}")
                    raise AssertionError("投机/普通greedy分歧；先定位同历史数值与位置，不放宽")

        result = generate_speculative(target, draft, ids, draft_tokens=k, max_new_tokens=bound,
                                      ignore_eos=ignore_eos, observer=observe)
        assert result["output_ids"] == reference[:bound]
        assert result["final_kv_length"] == ids.shape[1] + len(result["output_ids"]) - 1
        assert result["proposed"] == sum(len(r["proposals"]) for r in result["rounds"])
        assert result["accepted"] == sum(r["accepted"] for r in result["rounds"])
        assert result["verification_rounds"] == len(result["rounds"]) and result["recovered"]
        print(f"[PASS] K={k} 输出上限={bound}，完整序列={len(result['output_ids'])}Token与普通greedy一致，"
              f"提议/接受={result['proposed']}/{result['accepted']}，验证轮={result['verification_rounds']}，"
              f"接受长度={[r['accepted'] for r in result['rounds']]}，停止={result['stop_reason']}；两套KV/地址/回收通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--prompt", default="hi")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if not args.prompt.strip() or not 1 <= args.max_new_tokens <= 64:
        parser.error("输入不能为空，输出上限须为1至64")
    check_prefix()
    check_truncate()
    if args.model:
        target, draft, tokenizer = load_pair()
        ids = encode_prompt(tokenizer, args.prompt)
        check_draft_hf(draft, ids)
        check_verification(target, ids)
        check_generation(target, draft, ids, args.max_new_tokens, args.ignore_eos)
    print("[完成] 投机解码模型检查" if args.model else "[完成] CPU投机前缀与分页截断小用例")


if __name__ == "__main__":
    main()
