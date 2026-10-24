"""请求采样的独立参考与位置复现；--model 运行模型集成检查。"""

import argparse
import math
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from ampere_kv.sampler import (RequestSampler, distribution, sample_batch, sample_uniform,
                               cuda_topk, cuda_candidates)


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


def boundary_intervals(left, right, uniform, a, b):
    intervals = [(i, min(left[i], right[i]), max(left[i], right[i])) for i in range(min(a, b), max(a, b))]
    assert all(lo <= uniform < hi for _, lo, hi in intervals), (uniform, a, b, intervals)
    return intervals


def check_candidates(values, ids, temperatures, uniforms):
    """仅供检查：独立FP64概率、累计概率与各自区间选词。"""
    k = values.shape[1]
    out = torch.empty(values.shape[0], device=values.device, dtype=torch.long)
    probs, cdf = cuda_candidates(values, ids, temperatures, uniforms, out)
    reference_tokens, stats = [], dict(samples=0, boundary=0, max_cdf_abs=0.0, intervals=[])
    for row, (scores, t, u) in enumerate(zip(values, temperatures.tolist(), uniforms.tolist())):
        _, independent, _ = reference(scores, t, -1, u)
        _, ref_probs = distribution(scores, t, -1)
        ref_cdf = ref_probs.cumsum(0).clamp_max_(1); ref_cdf[-1] = 1
        p, q = ref_probs.tolist(), probs[row].tolist()
        left, right = ref_cdf.tolist(), cdf[row].tolist()
        truth = [min(1.0, math.fsum(independent[:i + 1])) for i in range(k)]; truth[-1] = 1.0
        accumulated = [min(1.0, math.fsum(q[:i + 1])) for i in range(k)]; accumulated[-1] = 1.0
        for actual in (p, q):
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(independent), rtol=2e-6, atol=1e-7)
        for actual in (left, right):
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in actual) and actual[-1] == 1
            assert all(a <= b for a, b in zip(actual, actual[1:]))
        error = max(abs(a - b) for x, y in ((left, truth), (right, truth), (left, right), (right, accumulated))
                    for a, b in zip(x, y))
        assert error <= 1e-6, ("CDF绝对误差", error)
        a = next((i for i, x in enumerate(left) if x > u), k - 1)
        b = next((i for i, x in enumerate(right) if x > u), k - 1)
        ref_token = sample_uniform(scores, t, -1, uniforms[row])
        assert ref_token.item() == a and out[row].item() == ids[row, b].item(), ("选词不符合自身CDF", row, u, a, b)
        reference_tokens.append(ids[row, a])
        if a != b:
            stats["intervals"].append((row, u, boundary_intervals(left, right, u, a, b)))
            stats["boundary"] += 1
        stats["samples"] += 1
        stats["max_cdf_abs"] = max(stats["max_cdf_abs"], error)
    return torch.stack(reference_tokens), out, stats


def check_sampling(logits, temperatures, uniforms, k):
    values, ids = cuda_topk(logits, k)
    scores, ordered = torch.sort(logits.float(), descending=True, stable=True)
    assert torch.equal(ids, ordered[:, :k]), "候选ID/顺序不同"
    torch.testing.assert_close(values, scores[:, :k], rtol=0, atol=0)
    return check_candidates(values, ids, temperatures, uniforms)


@contextmanager
def capture_sampling():
    """诊断运行临时捕获同一份采样输入；退出后恢复，不用于正式计时。"""
    from ampere_kv import scheduler, async_decode
    single, batch = RequestSampler.sample, sample_batch
    trace = dict(draws={}, outputs={})
    def remember(state, logits, position, token, backend):
        trace["draws"][state.seed, position] = (logits.clone(), state.uniforms[position].clone()
            if state.uniforms is not None else None, state.temperature, state.top_k, backend, token.clone())
    def sampled(state, logits, position, **kwargs):
        token = single(state, logits, position, **kwargs)
        backend = kwargs.get("sampling_backend") or state.backend
        if logits.numel() > 524288:
            backend = "reference"
        remember(state, logits, position, token, backend if state.top_k in (40, 128) and state.temperature > 0 else "reference")
        return token
    def batched(logits, requests, positions=None, out=None):
        tokens = batch(logits, requests, positions, out)
        states = [r.sampler for r in requests]
        fast = len(states) in (1, 4) and logits.shape[1] <= 524288 and states[0].top_k in (40, 128) and all(
            s.backend == "cuda" and s.top_k == states[0].top_k and s.temperature > 0 for s in states)
        for i, (r, p) in enumerate(zip(requests, positions if positions is not None else [len(r.output_ids) for r in requests])):
            remember(r.sampler, logits[i], p, tokens[i], "cuda" if fast else "reference")
        return tokens
    retire = scheduler.Scheduler._retire
    def retired(s, r):
        trace["outputs"][r.sampler.seed] = list(r.output_ids)
        return retire(s, r)
    RequestSampler.sample, scheduler.sample_batch, async_decode.sample_batch = sampled, batched, batched
    scheduler.Scheduler._retire = retired
    try:
        yield trace
    finally:
        RequestSampler.sample, scheduler.sample_batch, async_decode.sample_batch = single, batch, batch
        scheduler.Scheduler._retire = retire


def compare_traces(before, after):
    """每请求只检查首次分叉；后续不同历史不再作采样误差比较。"""
    boundaries = []
    for seed, expected in before["outputs"].items():
        actual = after["outputs"][seed]
        if expected == actual:
            continue
        position = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
        assert position < min(len(expected), len(actual)), "相同前缀却提前停止"
        x, u, t, k, backend, token = before["draws"][seed, position]
        y, v, other_t, other_k, other_backend, other_token = after["draws"][seed, position]
        assert u is not None and v is not None, "greedy首次选词不同"
        assert torch.equal(x, y) and torch.equal(u, v) and (t, k) == (other_t, other_k), "首次分歧输入不同，需定位模型路径"
        assert backend != other_backend, "相同采样后端首次选词不同"
        ref, cuda, stats = check_sampling(x[None], x.new_tensor([t], dtype=torch.float32), u[None], k)
        choices = {"reference": ref.item(), "cuda": cuda.item()}
        assert token.item() == expected[position] == choices[backend]
        assert other_token.item() == actual[position] == choices[other_backend] and stats["boundary"] == 1
        record = dict(seed=seed, position=position, backends=[backend, other_backend], intervals=stats["intervals"])
        boundaries.append(record)
        print(f"[边界] 首次分叉={record}；该请求后续历史不再对比")
    return boundaries


def check_cuda():
    from transformers import AutoConfig
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION
    vocab = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).vocab_size
    generator = torch.Generator(device="cuda").manual_seed(99)
    summary = {name: dict(samples=0, boundary=0, max_cdf_abs=0.0) for name in ("random", "boundary")}
    for batch in (1, 4):
        for k in (40, 128):
            for dtype in (torch.float32, torch.bfloat16):
                for size in (vocab, vocab + 13):
                    x = torch.randn(batch, size, device="cuda", generator=generator).to(dtype)
                    if size != vocab:
                        x[:, -13:] = 8  # 强制非整段尾部进入Top-K。
                    for kind in ("random", "ties", "zero", "negative", "extreme"):
                        if kind == "ties":
                            x.fill_(-2); x[:, ::17] = 1  # 第K名同分跨多个分段。
                        elif kind == "zero":
                            x.zero_(); x[:, ::2] = -0.0
                        elif kind == "negative":
                            x = -torch.arange(size, device="cuda", dtype=torch.float32)[None].expand(batch, -1).contiguous().to(dtype)
                        elif kind == "extreme":
                            x.fill_(-3e38); x[:, ::4096] = 3e38
                        values, ids = cuda_topk(x, k)
                        scores, ordered = torch.sort(x.float(), descending=True, stable=True)
                        assert torch.equal(ids, ordered[:, :k]), (batch, k, dtype, size, kind)
                        torch.testing.assert_close(values, scores[:, :k], rtol=0, atol=0)
                        temperatures = torch.linspace(0.6, 1.2, batch, device="cuda")
                        expected = [distribution(row, t.item(), -1)[1] for row, t in zip(values, temperatures)]
                        cdfs = [p.cumsum(-1).clamp_max_(1) for p in expected]
                        for c in cdfs:
                            c[-1] = 1
                        reference_cdf = torch.stack(cdfs)
                        edge = reference_cdf[:, k // 2]
                        uniforms = [torch.zeros_like(edge), torch.full_like(edge, 1 - 2**-24),
                                    torch.nextafter(edge, torch.zeros_like(edge)), edge,
                                    torch.nextafter(edge, torch.ones_like(edge))]
                        out = torch.empty(batch, device="cuda", dtype=torch.long)
                        _, other = cuda_candidates(values, ids, temperatures, uniforms[0], out)
                        other_edge = other[:, k // 2]
                        uniforms += [other_edge, torch.nextafter(other_edge, torch.zeros_like(edge)),
                                     torch.nextafter(other_edge, torch.ones_like(edge)), (edge + other_edge) * 0.5]
                        for name, u in [("boundary", u) for u in uniforms] + [
                                ("random", torch.rand(batch, device="cuda", generator=generator)) for _ in range(4)]:
                            try:
                                _, tokens, stats = check_candidates(values, ids, temperatures, u)
                            except AssertionError as error:
                                raise AssertionError((batch, k, dtype, size, kind, name, u.tolist())) from error
                            _, repeated, _ = check_candidates(values, ids, temperatures, u)
                            assert torch.equal(tokens, repeated), "同后端重复选词不同"
                            for key in ("samples", "boundary"):
                                summary[name][key] += stats[key]
                            summary[name]["max_cdf_abs"] = max(summary[name]["max_cdf_abs"], stats["max_cdf_abs"])
                        assert torch.equal(cuda_topk(x, k)[1], ids)
    for bad in (torch.zeros(1, 127, device="cuda"), torch.zeros(1, 256, device="cuda", dtype=torch.float16)):
        try:
            cuda_topk(bad, 128)
        except RuntimeError:
            pass
        else:
            raise AssertionError("扩展未拒绝非法词表长度或dtype")
    for k in (40, 128):
        states = [RequestSampler(8, vocab, "cuda", temperature=0.8, top_k=k, seed=i,
                                 sampling_backend="cuda") for i in range(4)]
        requests = [SimpleNamespace(sampler=s, output_ids=[]) for s in states]
        x = torch.zeros(4, vocab, device="cuda")
        tokens = sample_batch(x, requests, [0, 1, 2, 3])
        order = [3, 1, 0, 2]
        assert torch.equal(sample_batch(x[order], [requests[i] for i in order], [3, 1, 0, 2]), tokens[order])
        assert sample_batch(x[:1], requests[:1], [0])[0].item() == tokens[0].item()
        assert torch.equal(sample_batch(x, requests, [0, 1, 2, 3]), tokens)
        states[0].temperature = 0
        assert torch.equal(sample_batch(x, requests), torch.stack([r.sampler.sample(row, 0, sampling_backend="reference")
                                                                  for row, r in zip(x, requests)]))
    print("[PASS] CUDA候选、概率/CDF边界、尾部/同分与请求位置检查通过")
    print(f"[汇总] 普通随机数与主动边界样本分开统计：{summary}")


def check_model(sampling_backend="reference"):
    from ampere_kv.check_async_decode import CheckedScheduler, create, finish, check_stop_and_reuse
    from ampere_kv.runner import load_model_and_tokenizer

    class RecordingScheduler(CheckedScheduler):
        def __init__(self, *args, backend="reference", top_k=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.backend, self.top_k = backend, top_k

        def submit(self, request_id, *args, **kwargs):
            # 固定用例按完整ID选参数；replacement不是编号请求。
            index = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}.get(request_id, 4)
            temperature, k = (((0.8, self.top_k),) * 5 if self.top_k is not None else
                              ((0.7, 7), (1.2, 23), (0.0, -1), (0.9, -1), (1.0, 11)))[index]
            if self.force_eos and index == 0:
                k = 1  # 受控生命周期沿用greedy历史，避免随机EOS早于注入点。
            return super().submit(request_id, *args, **kwargs, temperature=temperature, top_k=k, seed=101 + index,
                                  sampling_backend=self.backend)

    model, _ = load_model_and_tokenizer()
    cases = [((15,), (8,), None, False, 40), ((15, 16, 63, 64), (5, 8, 6, 9), None, False, 128),
             ((15,), (8,), 16, False, 128), ((15, 16, 63, 64), (8, 8, 8, 8), None, True, 40)]
    if sampling_backend == "cuda":
        cases.append(((15, 16, 63, 64), (4, 4, 4, 4), None, False, 40))
    for lengths, limits, capacity, force_eos, top_k in cases:
        baselines = {}
        variants = [(backend, mode) for backend in (("reference", "cuda") if sampling_backend == "cuda" else ("reference",))
                    for mode in "abc"]
        for backend, mode in variants:
            def make_scheduler(*args, **kwargs):
                return RecordingScheduler(*args, backend=backend, top_k=top_k if sampling_backend == "cuda" else None, **kwargs)
            with capture_sampling() as trace:
                s, requests = create(model, mode, lengths, limits, capacity=capacity, force_eos=force_eos,
                                     scheduler_type=make_scheduler)
                if mode == "c":
                    s.decode_pipeline.advance()
                    s.decode_pipeline.drain()
                batches = finish(s, requests)
            output = [(r.output_ids, r.stop_reason) for r in requests]
            if backend not in baselines:
                if backend == "cuda":
                    compare_traces(baselines["reference"][2], trace)
                baselines[backend] = s, output, trace
            baseline = baselines[backend]
            compare_traces(baseline[2], trace)
            for i, (actual, expected) in enumerate(zip(output, baseline[1])):
                if actual[0] == expected[0]:
                    assert actual[1] == expected[1], "相同输出却停止原因不同"
                    if not force_eos:
                        for a_layer, b_layer in zip(s.saved[f"r{i}"], baseline[0].saved[f"r{i}"]):
                            for a, b in zip(a_layer, b_layer):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for r in requests:
                assert r.sampler.uniforms is None
                if r.sampler.temperature > 0:
                    assert all((r.sampler.seed, p) in trace["draws"] for p in range(len(r.output_ids)))
            if capacity:
                assert s.graph_fallbacks > 0
            if len(set(limits)) > 1:
                assert {2, 3} <= batches
            if force_eos:
                assert requests[0].stop_reason == "EOS" and len(requests[0].output_ids) == 3
                if mode == "c":
                    assert s.decode_pipeline.stats["discarded_tokens"] >= 1
            k_label = top_k if sampling_backend == "cuda" else "逐请求"
            print(f"[PASS] 采样后端={backend} 路径={mode} K={k_label} 历史={lengths}：输出/停止/KV及释放；批量={sorted(batches)}")
    check_stop_and_reuse(model, scheduler_type=make_scheduler)
    print("[完成] 请求级采样与模型集成检查通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="纯采样通过后另跑短模型与生命周期")
    parser.add_argument("--sampling-backend", choices=("reference", "cuda"), default="reference")
    args = parser.parse_args()
    check_probabilities("cuda")
    check_positions("cuda")
    if args.sampling_backend == "cuda":
        check_cuda()
    if args.model:
        check_model(args.sampling_backend)


if __name__ == "__main__":
    main()
