"""请求采样的独立参考与位置复现；--model 运行模型集成检查。"""

import argparse
import math
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


def check_cuda():
    from transformers import AutoConfig
    from ampere_kv.runner import MODEL_ID, MODEL_REVISION
    vocab = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).vocab_size
    generator = torch.Generator(device="cuda").manual_seed(99)
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
                        for u in uniforms:
                            probs, cdf = cuda_candidates(values, ids, temperatures, u, out)
                            for row in range(batch):
                                _, independent, _ = reference(values[row], temperatures[row].item(), -1, u[row].item())
                                torch.testing.assert_close(probs[row].cpu(), torch.tensor(independent), rtol=2e-6, atol=1e-7)
                            choices = torch.searchsorted(reference_cdf.contiguous(), u[:, None], right=True).clamp_max(k - 1)
                            tokens = ids.gather(1, choices).flatten()
                            if not torch.equal(out, tokens):
                                print(f"[分歧] B={batch} K={k} V={size} dtype={dtype} case={kind} uniform={u.tolist()}")
                                print(f"[分歧] 候选={ids.tolist()} CDF差={(cdf-reference_cdf).tolist()} "
                                      f"reference={tokens.tolist()} cuda={out.tolist()}")
                                raise AssertionError("候选一致但CDF边界选词不同")
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


def check_model(sampling_backend="reference"):
    from ampere_kv.check_async_decode import CheckedScheduler, create, finish, check_stop_and_reuse
    from ampere_kv.runner import load_model_and_tokenizer

    class RecordingScheduler(CheckedScheduler):
        def __init__(self, *args, backend="reference", top_k=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.draws = {}
            self.parameters = {}
            self.cache_requests = {}
            self.backend, self.top_k = backend, top_k

        def enable_graphs(self, max_tokens):
            super().enable_graphs(max_tokens)
            for graph in self.graphs.values():
                for name in ("step", "step_device"):
                    original = getattr(graph, name)

                    def recorded_graph(caches, *args, original=original):
                        logits = original(caches, *args)
                        for row, cache in enumerate(caches[0]):
                            r = self.cache_requests[id(cache)]
                            self.remember(r, logits[row, 0], cache.length - r.input_ids.shape[1])
                        return logits

                    setattr(graph, name, recorded_graph)

        def _admit(self):
            super()._admit()
            if self.prefilling is not None:
                self.cache_requests[id(self.prefilling.caches[0])] = self.prefilling

        def remember(self, request, logits, position):
            state = request.sampler
            self.draws[request.request_id][position] = (logits.detach().clone(),
                state.uniforms[position].clone() if state.uniforms is not None else None, None)

        def submit(self, request_id, *args, **kwargs):
            # 固定用例按完整ID选参数；replacement不是编号请求。
            index = {"r0": 0, "r1": 1, "r2": 2, "r3": 3}.get(request_id, 4)
            temperature, k = (((0.8, self.top_k),) * 5 if self.top_k is not None else
                              ((0.7, 7), (1.2, 23), (0.0, -1), (0.9, -1), (1.0, 11)))[index]
            if self.force_eos and index == 0:
                k = 1  # 受控生命周期沿用greedy历史，避免随机EOS早于注入点。
            request = super().submit(request_id, *args, **kwargs, temperature=temperature, top_k=k, seed=101 + index,
                                     sampling_backend=self.backend)
            self.parameters[request_id] = temperature, k
            state, draws = request.sampler, self.draws.setdefault(request_id, {})
            sample = state.sample

            def recorded(logits, position, **kwargs):
                token = sample(logits, position, **kwargs)
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
            print(f"[分歧] {label} uniform={None if uniform is None else uniform.item()} Token={None if token is None else token.item()} "
                  f"前8候选={ids[:8].tolist()} 概率={probs[:8].tolist()}")

    model, _ = load_model_and_tokenizer()
    cases = [((15,), (8,), None, False, 40), ((15, 16, 63, 64), (5, 8, 6, 9), None, False, 128),
             ((15,), (8,), 16, False, 128), ((15, 16, 63, 64), (8, 8, 8, 8), None, True, 40)]
    if sampling_backend == "cuda":
        cases.append(((15, 16, 63, 64), (4, 4, 4, 4), None, False, 40))
    for lengths, limits, capacity, force_eos, top_k in cases:
        baseline = None
        variants = [(backend, mode) for backend in (("reference", "cuda") if sampling_backend == "cuda" else ("reference",))
                    for mode in "abc"]
        for backend, mode in variants:
            def make_scheduler(*args, **kwargs):
                return RecordingScheduler(*args, backend=backend, top_k=top_k if sampling_backend == "cuda" else None, **kwargs)
            s, requests = create(model, mode, lengths, limits, capacity=capacity, force_eos=force_eos,
                                 scheduler_type=make_scheduler)
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
                        print(f"[分歧] 输出reference={expected[0]} {backend}/{mode}={actual[0]}")
                        diagnose(baseline[0], s, f"r{i}", position, f"{backend}/{mode}")
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
