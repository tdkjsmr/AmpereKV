"""固定模型的单步流水对照；受控EOS/取消仅验证生命周期，不冒充模型精度。"""

import time

import torch

from ampere_kv.runner import encode_prompt, load_model_and_tokenizer
from ampere_kv.scheduler import RUNNING, Scheduler


class CheckedScheduler(Scheduler):
    def __init__(self, *args, force_eos=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved, self.released = {}, {}
        self.force_eos = force_eos

    def _commit(self, request, token):
        if self.force_eos and request.request_id == "r0" and len(request.output_ids) == 2:
            token = next(iter(self.eos_ids))
        super()._commit(request, token)

    def _retire(self, request):
        p = self.decode_pipeline
        assert p is None or p.pending is None, "设备仍引用在途缓存时退休"
        if request.caches:
            valid = request.input_ids.shape[1] + max(0, len(request.output_ids) - 1)
            self.saved[request.request_id] = [tuple(t[:, :, :valid].clone() for t in c.get()) for c in request.caches]
            self.released[request.request_id] = set(request.caches[0]._table.block_ids)
        super()._retire(request)


def create(model, mode, lengths, limits, *, force_eos=False, natural_eos=False, capacity=None, prompts=None,
           scheduler_type=CheckedScheduler):
    prompts = prompts or [torch.arange(1, n + 1, device="cuda", dtype=torch.long)[None] for n in lengths]
    blocks = sum((p.shape[1] + limit + 16) // 16 for p, limit in zip(prompts, limits))
    s = scheduler_type(model, blocks, max_batch=len(limits), force_eos=force_eos)
    maximum = capacity or (max(p.shape[1] + n for p, n in zip(prompts, limits)) + 15) // 16 * 16
    s.enable_graphs(maximum)
    if mode != "a":
        s.enable_pipeline(overlap=mode == "c")
    requests = [s.submit(f"r{i}", p, n, ignore_eos=not (force_eos or natural_eos))
                for i, (p, n) in enumerate(zip(prompts, limits))]
    # 测试固定Decode组：只复用原Prefill入口，不在初始化时推进邻座Decode。
    for r in requests:
        s._admit()
        assert s.prefilling is r
        s._advance(r, prefill=True)
        s.prefilling = None
        r.status = RUNNING
        s.running.append(r)
    return s, requests


def check_host_state(s):
    for layer in range(len(s.storages)):
        ownership = [set(r.caches[layer]._table.block_ids) for r in s.running]
        assert sum(map(len, ownership)) == len(set().union(*ownership))
        for r in s.running:
            committed = r.input_ids.shape[1] + len(r.output_ids) - 1
            assert r.caches[layer].length - committed in (0, 1), "返回时领先超过一轮"


def finish(s, requests):
    finished_ids, batches = set(), set()
    for _ in range(256):
        if all(r.status == "finished" for r in requests):
            break
        done, sizes = s.decode_pipeline.advance() if s.decode_pipeline else s.step()
        for r in done:
            assert r.request_id not in finished_ids, "重复返回已完成请求"
            finished_ids.add(r.request_id)
        batches.update(sizes)
        check_host_state(s)
    else:
        raise AssertionError("未在有限轮次内结束")
    assert s.reserved_blocks == 0 and not s.waiting and not s.running and s.prefilling is None
    assert all(storage._pool.num_free_blocks == s.blocks_per_layer for storage in s.storages)
    if s.decode_pipeline:
        p = s.decode_pipeline
        assert p.pending is None and p.stats["max_inflight"] <= 2
        assert p.stats["device_tokens"] == p.stats["committed_tokens"] + p.stats["discarded_tokens"]
    return batches


def compare(model, lengths, limits, *, force_eos=False, natural_eos=False, capacity=None, prompts=None):
    reference = None
    for mode in "abc":
        s, requests = create(model, mode, lengths, limits, force_eos=force_eos,
                             natural_eos=natural_eos, capacity=capacity, prompts=prompts)
        pointers = [(x._key.data_ptr(), x._value.data_ptr()) for x in s.storages]
        slots = [slot for pair in s.decode_pipeline.slots.values() for slot in pair] if s.decode_pipeline else []
        buffers = [(slot.sampled_tokens.data_ptr(), slot.output_host.data_ptr(), slot.metadata_host.data_ptr())
                   for slot in slots]
        batches = finish(s, requests)
        output = [(r.output_ids, r.stop_reason) for r in requests]
        assert pointers == [(x._key.data_ptr(), x._value.data_ptr()) for x in s.storages]
        assert buffers == [(slot.sampled_tokens.data_ptr(), slot.output_host.data_ptr(), slot.metadata_host.data_ptr())
                           for slot in slots]
        if natural_eos:
            assert requests[0].stop_reason == "EOS", "本次未覆盖真实EOS，不计为EOS通过"
        if capacity:
            assert s.graph_fallbacks > 0, "未走到Graph容量回退"
        if len(limits) == 4 and len(set(limits)) > 1:
            assert {2, 3} <= batches, "未覆盖B2/B3退出回退"
        if reference is None:
            reference = output, s.saved
        else:
            assert output == reference[0], f"{mode}序列或停止原因不同：{output} != {reference[0]}"
            if not force_eos:
                for rid, layers in s.saved.items():
                    for actual, expected in zip(layers, reference[1][rid]):
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if force_eos:
            assert requests[0].stop_reason == "EOS" and len(requests[0].output_ids) == 3
            if mode == "c":
                assert s.decode_pipeline.stats["discarded_tokens"] >= 1
        print(f"[PASS] 路径={mode} 历史={lengths} 上限={limits}：输出/停止与A一致，块全归还；批量={sorted(batches)}")
        if mode != "a":
            print(f"[流水] {dict(s.decode_pipeline.stats)}")


def check_stop_and_reuse(model, scheduler_type=CheckedScheduler):
    for mode in "bc":
        s, requests = create(model, mode, (15, 16, 63, 64), (8, 8, 8, 12), scheduler_type=scheduler_type)
        p = s.decode_pipeline
        p.advance()
        if mode == "c":
            try:
                s.step()
            except RuntimeError:
                pass
            else:
                raise AssertionError("在途期间原step未拒绝混用")
        before = [len(r.output_ids) for r in requests]
        held = set().union(*(set(r.caches[0]._table.block_ids) for r in requests[:3]))
        s.cancel(requests[0])
        s.cancel(requests[1])
        requests[1].deadline = requests[2].deadline = time.monotonic() - 1
        done, _ = p.advance()
        assert {r.request_id for r in done} == {"r0", "r1", "r2"}
        assert [r.stop_reason for r in requests[:3]] == ["已取消", "已取消", "已超时"]
        assert [len(r.output_ids) for r in requests[:3]] == before[:3]
        assert p.pending is None
        replacement = s.submit("replacement", requests[0].input_ids, 5, ignore_eos=True)
        requests.append(replacement)
        seen = set()
        for _ in range(256):
            if all(r.status == "finished" for r in requests):
                break
            p.advance()
            if replacement.caches:
                seen.update(replacement.caches[0]._table.block_ids)
            check_host_state(s)
        else:
            raise AssertionError("替换请求未完成")
        assert seen & held, "未复用归还的旧块"
        assert len(requests[3].output_ids) == 12 and len(replacement.output_ids) == 5
        finish(s, requests)
        print(f"[PASS] {mode}：取消/到期优先级、在途丢弃后安全回收、替换请求块复用和邻座完成")


def main():
    model, tokenizer = load_model_and_tokenizer()
    compare(model, (15,), (8,))
    compare(model, (15, 16, 63, 64), (5, 8, 6, 9))
    compare(model, (15,), (8,), capacity=16)
    # CPU选词入口注入EOS只检验投机丢弃；其他真实模型greedy对照不注入。
    compare(model, (15, 16, 63, 64), (8, 8, 8, 8), force_eos=True)
    prompt = encode_prompt(tokenizer, "hi")
    compare(model, (prompt.shape[1],), (32,), prompts=[prompt], natural_eos=True)
    check_stop_and_reuse(model)
    s, requests = create(model, "c", (15,), (8,))
    s.decode_pipeline.advance()
    before = len(requests[0].output_ids)
    s.decode_pipeline.drain()
    assert s.decode_pipeline.pending is None and len(requests[0].output_ids) == before + 1
    finish(s, requests)
    print("[PASS] 显式drain提交最后在途结果，随后可重入；跨块、序列及缓冲轮换已覆盖")
    print("[完成] 单步流水正确性检查通过")


if __name__ == "__main__":
    main()
