"""请求采样的独立参考与位置复现；--model 运行模型集成检查。"""

import argparse
import math
from types import SimpleNamespace

import torch

from ampere_kv.sampler import RequestSampler, distribution, sample_batch, sample_uniform


def reference(logits, temperature, top_k, uniform):
    values = logits.float().cpu().tolist()
    ids = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ids = ids if top_k == -1 else ids[:top_k]
    weights = [math.exp((values[i] - values[ids[0]]) / temperature) for i in ids]
    probabilities = [w / sum(weights) for w in weights]
    total = 0.0
    for token, probability in zip(ids, probabilities):
        total += probability
        if uniform < total:
            return ids, probabilities, token
    return ids, probabilities, ids[-1]


def check_probabilities(device):
    cases = ([2.0, 0.0, -1.0, 2.0], [0.0] * 4, [3e38, -3e38, 0.0, 3e38])
    for values in cases:
        logits = torch.tensor(values, device=device)
        for temperature in (0.25, 1.0, 2.0):
            for k in (1, 2, 4, -1):
                for uniform in (0.0, 0.125, 0.375, 0.875, 1 - 2**-24):
                    ids, probs, expected = reference(logits, temperature, k, uniform)
                    actual_ids, actual_probs = distribution(logits, temperature, k)
                    assert actual_ids.tolist() == ids
                    torch.testing.assert_close(actual_probs.cpu(), torch.tensor(probs), rtol=2e-6, atol=1e-7)
                    actual = sample_uniform(logits, temperature, k, torch.tensor(uniform, device=device))
                    assert actual.item() == expected, (values, temperature, k, uniform, actual, expected)
    tied = torch.zeros(4, device=device)
    for k in (1, 2, -1):
        assert distribution(tied, 1.0, k)[0].tolist() == list(range(4 if k == -1 else k))
    assert sample_uniform(tied, 1.0, -1, torch.tensor(0.25, device=device)).item() == 1
    assert sample_uniform(tied, 1.0, 2, torch.tensor(0.5, device=device)).item() == 1
    assert sample_uniform(tied, 1.0, 1, torch.tensor(0.99, device=device)).item() == tied.argmax().item() == 0
    print("[PASS] 独立FP64概率/逆CDF参考、温度/top-k/全词表、同分边界与极端有限logits")


def check_positions(device):
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    params = dict(temperature=0.8, top_k=3, seed=73)
    state = RequestSampler(12, 5, device, **params)
    twin = RequestSampler(12, 5, device, **params)
    shorter = RequestSampler(6, 5, device, **params)
    other = RequestSampler(12, 5, device, temperature=1.2, top_k=-1, seed=74)
    automatic = RequestSampler(12, 5, device, temperature=1)
    replay = RequestSampler(12, 5, device, temperature=1, seed=automatic.seed)
    assert torch.equal(state.uniforms, twin.uniforms) and not torch.equal(state.uniforms, other.uniforms)
    assert torch.equal(state.uniforms[:6], shorter.uniforms)
    assert torch.equal(automatic.uniforms, replay.uniforms)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    row = torch.tensor([2.0, 2.0, 2.0, -1.0, -2.0], device=device)
    requests = [SimpleNamespace(sampler=s, output_ids=[0] * n)
                for s, n in ((state, 0), (other, 4), (twin, 1))]
    logits = torch.stack((row, row.flip(0), row))
    baseline = sample_batch(logits, requests)
    permutation = (2, 0, 1)
    permuted = sample_batch(logits[list(permutation)], [requests[i] for i in permutation])
    assert torch.equal(permuted, baseline[list(permutation)])
    assert torch.equal(sample_batch(logits[:1], requests[:1]), baseline[:1])
    for position in range(8):
        other.sample(row, position)  # 无关请求前进不应消耗本请求的随机数。
    assert torch.equal(sample_batch(logits, requests), baseline)
    for position in (0, 1, 5, 6, 5, 1):
        assert torch.equal(state.sample(row, position), twin.sample(row, position))
        ids, probs, token = reference(row, state.temperature, state.top_k, state.uniforms[position].item())
        assert state.sample(row, position).item() == token
        assert distribution(row, state.temperature, state.top_k)[0].tolist() == ids
    # 模拟CPU只提交到位置0，而设备已经发射位置2；显式位置优先于output_ids。
    explicit = sample_batch(logits[:1], requests[:1], [2])
    assert explicit[0].item() == state.sample(row, 2).item()
    greedy = RequestSampler(12, 5, device)
    assert greedy.uniforms is None and greedy.sample(row, 7).item() == row.argmax().item()
    greedy_requests = [SimpleNamespace(sampler=greedy) for _ in range(3)]
    assert torch.equal(sample_batch(logits, greedy_requests), logits.argmax(dim=-1))
    for seed in (0, 2**32 - 1):
        assert RequestSampler(2, 5, device, temperature=1, seed=seed).seed == seed
    out = torch.empty(1, dtype=torch.long, device=device)
    assert sample_batch(logits[:1], requests[:1], [1], out=out) is out
    assert out[0].item() == state.sample(row, 1).item()
    for invalid in (dict(temperature=-1), dict(temperature=float("nan")), dict(temperature=float("inf")),
                    dict(top_k=0), dict(top_k=6), dict(top_k=True), dict(seed=-1), dict(seed=2**32)):
        try:
            RequestSampler(12, 5, device, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"未拒绝非法参数：{invalid}")
    state.release()
    assert state.uniforms is None
    print("[PASS] 请求私有uniform/全局RNG不变、seed复现、行重排/插入退出、位置0/1与前瞻丢弃重取")


def check_model():
    from ampere_kv.check_async_decode import CheckedScheduler, create, finish, check_stop_and_reuse
    from ampere_kv.runner import load_model_and_tokenizer

    class RecordingScheduler(CheckedScheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.draws = {}
            self.parameters = {}

        def submit(self, request_id, *args, **kwargs):
            # 固定用例按完整ID选参数；replacement不是编号请求。
            index = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}.get(request_id, 4)
            temperature, k = ((0.7, 7), (1.2, 23), (0.0, -1), (0.9, -1), (1.0, 11))[index]
            if self.force_eos and index == 0:
                k = 1  # 受控生命周期沿用greedy历史，避免随机EOS早于注入点。
            request = super().submit(request_id, *args, **kwargs, temperature=temperature, top_k=k, seed=101 + index)
            self.parameters[request_id] = temperature, k
            state, draws = request.sampler, self.draws.setdefault(request_id, {})
            sample = state.sample

            def recorded(logits, position):
                token = sample(logits, position)
                draws[position] = (logits.detach().clone(),
                                   state.uniforms[position].clone() if state.uniforms is not None else None,
                                   token.clone())
                return token

            state.sample = recorded
            return request

    def diagnose(baseline, current, rid, position, mode):
        before, after = baseline.draws[rid].get(position), current.draws[rid].get(position)
        print(f"[分歧] 路径={mode} 请求={rid} 逻辑输出位置={position}（首Token=0）")
        if before is None or after is None:
            print("[分歧] 该位置未执行采样或走整组greedy快路径，没有随机采样记录")
            return
        temperature, k = current.parameters[rid]
        print(f"[分歧] logits最大差={(before[0].float() - after[0].float()).abs().max().item()}")
        for label, (logits, uniform, token) in zip(("A", mode), (before, after)):
            ids, probs = distribution(logits, temperature or 1.0, k)
            print(f"[分歧] {label} uniform={None if uniform is None else uniform.item()} Token={token.item()} "
                  f"前8候选={ids[:8].tolist()} 概率={probs[:8].tolist()}")

    model, _ = load_model_and_tokenizer()
    cases = [((15,), (8,), None, False), ((15, 16, 63, 64), (5, 8, 6, 9), None, False),
             ((15,), (8,), 16, False), ((15, 16, 63, 64), (8, 8, 8, 8), None, True)]
    for lengths, limits, capacity, force_eos in cases:
        baseline = None
        for mode in "abc":
            s, requests = create(model, mode, lengths, limits, capacity=capacity, force_eos=force_eos,
                                 scheduler_type=RecordingScheduler)
            if mode == "c":
                s.decode_pipeline.advance()
                s.decode_pipeline.drain()
            batches = finish(s, requests)
            output = [(r.output_ids, r.stop_reason) for r in requests]
            if baseline is None:
                baseline = s, output
            else:
                for i, (actual, expected) in enumerate(zip(output, baseline[1])):
                    if actual != expected:
                        position = next((j for j, (a, b) in enumerate(zip(actual[0], expected[0])) if a != b),
                                        min(len(actual[0]), len(expected[0])))
                        diagnose(baseline[0], s, f"r{i}", position, mode)
                        raise AssertionError("同成员同历史随机对照分歧")
                if not force_eos:
                    for rid, layers in s.saved.items():
                        for actual, expected in zip(layers, baseline[0].saved[rid]):
                            for a, b in zip(actual, expected):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for r in requests:
                assert r.sampler.uniforms is None
                if r.sampler.temperature > 0:
                    assert set(range(len(r.output_ids))) <= s.draws[r.request_id].keys()
            if capacity:
                assert s.graph_fallbacks > 0
            if len(set(limits)) > 1:
                assert {2, 3} <= batches
            if force_eos:
                assert requests[0].stop_reason == "EOS" and len(requests[0].output_ids) == 3
                if mode == "c":
                    assert s.decode_pipeline.stats["discarded_tokens"] >= 1
            print(f"[PASS] 采样路径={mode} 历史={lengths}：位置0/Decode、输出/停止/KV及释放；批量={sorted(batches)}")
    check_stop_and_reuse(model, scheduler_type=RecordingScheduler)
    print("[完成] 请求级采样与模型集成检查通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="纯采样通过后另跑短模型与生命周期")
    args = parser.parse_args()
    check_probabilities("cuda")
    check_positions("cuda")
    if args.model:
        check_model()


if __name__ == "__main__":
    main()
