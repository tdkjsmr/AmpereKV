"""结构化输出：默认检查真实tokenizer/掩码，--model运行同步模型与生命周期演示。"""

import argparse
import json
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ampere_kv.runner import MODEL_ID, MODEL_REVISION, encode_prompt, load_model_and_tokenizer
from ampere_kv.sampler import RequestSampler, sample_batch, cuda_sample, sample_uniform
from ampere_kv.scheduler import FINISHED, RUNNING, Scheduler
from ampere_kv.structured_output import SchemaCompiler


CATEGORY = {"type": "object", "properties": {
    "category": {"type": "string", "enum": ["work", "personal", "other"]},
    "priority": {"type": "integer", "enum": [1, 2, 3]}},
    "required": ["category", "priority"], "additionalProperties": False}
APPROVAL = {"type": "object", "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"], "additionalProperties": False}
PATHS = ("eager-reference", "graph-reference", "graph-cuda")


def validate(text, schema):
    value = json.loads(text)
    assert type(value) is dict and set(value) == set(schema["required"])
    for name, rule in schema["properties"].items():
        assert type(value[name]) is {"string": str, "integer": int, "boolean": bool}[rule["type"]]
        if "enum" in rule:
            assert value[name] in rule["enum"]
    return value


def rejected(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("非法组合未被拒绝")


def allowed(state, vocab):
    logits = torch.zeros(vocab, device="cuda")
    state.mask(logits)
    return torch.isfinite(logits).nonzero().flatten().tolist()


def feed(state, tokenizer, text):
    for token in tokenizer.encode(text, add_special_tokens=False):
        assert not state.accept(token)


def check_masks(tokenizer, vocab, stops):
    subprocess.run([sys.executable, "-c", "import sys; sys.modules['xgrammar']=None; import ampere_kv.sampler"], check=True)
    compiler = SchemaCompiler(tokenizer, vocab, stops)
    rejected(lambda: compiler.compile({"type": "invalid-type"}))
    grammar = compiler.compile({"type": "string", "enum": ["A", "B"]})
    left, right = grammar.new_state("cuda"), grammar.new_state("cuda")
    initial = allowed(left, vocab)
    assert initial == allowed(right, vocab) and left.matcher is not right.matcher
    assert left.cpu_mask.data_ptr() != right.cpu_mask.data_ptr() and left.device_mask.data_ptr() != right.device_mask.data_ptr()
    actual_ids = set(tokenizer.get_vocab().values())
    assert set(initial) <= actual_ids
    illegal = next(i for i in range(vocab) if i not in set(initial))
    logits = torch.zeros(vocab, device="cuda"); logits[illegal] = 100
    sampler = RequestSampler(16, vocab, "cuda", structured=left)
    assert sampler.sample(logits, 0).item() in initial
    assert allowed(left, vocab) == initial  # 填掩码与选词都不推进matcher。
    feed(left, tokenizer, '"')
    choices = allowed(left, vocab)
    assert 1 < len(choices) < 40 and allowed(right, vocab) == initial
    sampler.release(); right.release()
    state = grammar.new_state("cuda")
    feed(state, tokenizer, '"')
    random = RequestSampler(16, vocab, "cuda", temperature=0.8, top_k=40, seed=42, structured=state)
    logits = torch.zeros(vocab, device="cuda"); logits[illegal] = 100
    assert random.sample(logits, 0).item() in choices
    random.release()

    single = SchemaCompiler(tokenizer, vocab, stops[:1]).compile({"type": "string", "enum": ["A"]})
    for temperature in (0.0, 0.8):
        state = single.new_state("cuda")
        feed(state, tokenizer, '"A"')
        assert allowed(state, vocab) == stops[:1]
        sampler = RequestSampler(16, vocab, "cuda", temperature=temperature, top_k=40, seed=43, structured=state)
        logits = torch.zeros(vocab, device="cuda"); logits[illegal] = 100
        request = SimpleNamespace(sampler=sampler, output_ids=[])
        token = sample_batch(logits[None], [request])[0].item()
        assert token == stops[0] and state.accept(token)
        sampler.release()
        assert state.matcher is state.compiled is state.cpu_mask is state.device_mask is None
    empty = single.new_state("cuda")
    empty.matcher = SimpleNamespace(fill_next_token_bitmask=lambda mask: mask.zero_())
    rejected(lambda: empty.mask(torch.zeros(vocab, device="cuda")))
    empty.release()
    print(f"[PASS] 真实tokenizer/模型词表={len(actual_ids)}/{vocab}，非法最高分、少于K/单候选、EOS、独立状态与零合法项拒绝")

    stub = SimpleNamespace(config=SimpleNamespace(vocab_size=vocab), model=SimpleNamespace(layers=[]),
                           generation_config=SimpleNamespace(eos_token_id=stops))
    scheduler = Scheduler(stub, 64)
    prompt = torch.tensor([[1]], device="cuda")
    for field, value in (("decode_pipeline", object()), ("mixed", True), ("prefill_chunk_size", 16)):
        previous = getattr(scheduler, field)
        setattr(scheduler, field, value)
        rejected(lambda: scheduler.submit("bad", prompt, 16, structured=grammar))
        setattr(scheduler, field, previous)
    rejected(lambda: scheduler.submit("bad", prompt, 16, structured=grammar, ignore_eos=True))
    assert not scheduler.waiting and not scheduler._submitted
    print("[PASS] 可选依赖不影响普通导入；结构化流水/mixed/分块/ignore_eos在提交边界拒绝")


def check_cuda_masks(tokenizer, vocab, stops):
    from ampere_kv.check_sampler import check_sampling, cumulative_probabilities, distribution

    compiler = SchemaCompiler(tokenizer, vocab, stops)
    cases = (({"type": "string", "enum": ["A"]}, '"A"', "eos"),
             ({"type": "string", "enum": ["A", "B"]}, '"', "few"),
             ({"type": "string"}, '"', "many"))
    generator = torch.Generator(device="cuda").manual_seed(91)
    totals = dict(samples=0, boundary=0, max_cdf_abs=0.0)
    for schema, prefix, kind in cases:
        grammar = (SchemaCompiler(tokenizer, vocab, stops[:1]) if kind == "eos" else compiler).compile(schema)
        for batch in (1, 4):
            for k in (40, 128):
                for dtype in (torch.float32, torch.bfloat16):
                    states = [grammar.new_state("cuda") for _ in range(batch)]
                    for state in states:
                        feed(state, tokenizer, prefix)
                    valid = allowed(states[0], vocab)
                    assert (len(valid) == 1 if kind == "eos" else
                            1 < len(valid) < 40 if kind == "few" else len(valid) > 128)
                    logits = torch.randn(batch, vocab, generator=generator, device="cuda").to(dtype)
                    logits[:, valid[:min(3, len(valid))]] = 2  # 保留合法同分，非法项设最高分。
                    valid_ids = set(valid)
                    illegal = next(i for i in range(vocab) if i not in valid_ids)
                    logits[:, illegal] = 100
                    masked = logits.clone()
                    for state, row in zip(states, masked):
                        state.mask(row)
                    temperatures = torch.full((batch,), 0.8, device="cuda")
                    for probe in (0.0, 0.71, 1 - 2**-24, None):
                        uniforms = torch.full((batch,), 0.0 if probe is None else probe, device="cuda")
                        if probe is None:
                            uniforms = torch.stack([cumulative_probabilities(distribution(row, 0.8, k)[1])[0]
                                                    for row in masked]).clamp_max(1 - 2**-24).contiguous()
                        _, expected, stats = check_sampling(masked, temperatures, uniforms, k)
                        totals["samples"] += stats["samples"]
                        totals["boundary"] += stats["boundary"]
                        totals["max_cdf_abs"] = max(totals["max_cdf_abs"], stats["max_cdf_abs"])
                        requests = [SimpleNamespace(output_ids=[], sampler=RequestSampler(
                            1, vocab, "cuda", temperature=0.8, top_k=k, seed=42 + i,
                            sampling_backend="cuda", structured=state)) for i, state in enumerate(states)]
                        for request, u in zip(requests, uniforms):
                            request.sampler.uniforms[0] = u
                        with patch("ampere_kv.sampler.cuda_sample", wraps=cuda_sample) as calls:
                            with_masks = [patch.object(state, "mask", wraps=state.mask) for state in states]
                            mocks = [wrapper.start() for wrapper in with_masks]
                            try:
                                output = sample_batch(logits.clone(), requests)
                                assert torch.equal(output, expected)
                                assert calls.call_count == 1 and calls.call_args.args[0].shape[0] == batch
                                assert all(mock.call_count == 1 for mock in mocks)
                            finally:
                                for wrapper in reversed(with_masks):
                                    wrapper.stop()
                        assert all(allowed(state, vocab) == valid for state in states)
                        if batch == 1:
                            with patch.object(states[0], "mask", wraps=states[0].mask) as mask:
                                assert requests[0].sampler.sample(logits[0].clone(), 0).item() == expected[0].item()
                                assert mask.call_count == 1
                    for state in states:
                        state.release()
    # 全greedy、混合greedy、异构K和B2均应逐行过滤一次，后两者回退reference。
    grammar = compiler.compile(cases[1][0])
    for parameters in (((0, 40),) * 4, ((0, 40), (0.8, 40), (0.8, 128), (0.8, 40)),
                       ((0.8, 40), (0.8, 128))):
        requests, masks = [], []
        for i, (temperature, k) in enumerate(parameters):
            state = grammar.new_state("cuda")
            feed(state, tokenizer, '"')
            requests.append(SimpleNamespace(output_ids=[], sampler=RequestSampler(
                2, vocab, "cuda", temperature=temperature, top_k=k, seed=42 + i,
                sampling_backend="cuda", structured=state)))
            masks.append(patch.object(state, "mask", wraps=state.mask))
        logits = torch.randn(len(requests), vocab, generator=generator, device="cuda")
        masked = logits.clone()
        for request, row in zip(requests, masked):
            request.sampler.structured.mask(row)
        expected = torch.stack([row.argmax() if r.sampler.temperature == 0 else
                                sample_uniform(row, r.sampler.temperature, r.sampler.top_k, r.sampler.uniforms[0])
                                for row, r in zip(masked, requests)])
        with patch("ampere_kv.sampler.cuda_sample", wraps=cuda_sample) as calls:
            mocks = [wrapper.start() for wrapper in masks]
            try:
                assert torch.equal(sample_batch(logits, requests), expected) and calls.call_count == 0
                assert all(mock.call_count == 1 for mock in mocks)
            finally:
                for wrapper in reversed(masks):
                    wrapper.stop()
        for request in requests:
            request.sampler.release()
    print(f"[PASS] 结构化B1/B4、K40/128、FP32/BF16候选/CDF、单项/少于K/多于K、同分与EOS；统计={totals}")
    print("[PASS] 批量CUDA实际命中；greedy/异构回退每行掩码一次，选词不推进matcher")


class AuditedScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states, self.accepted, self.retired = {}, {}, []
        self.masks, self.batches, self.fresh_logits = {}, [], 0

    def submit(self, request_id, *args, **kwargs):
        request = super().submit(request_id, *args, **kwargs)
        state = request.sampler.structured
        if state is not None:
            self.states[request_id], self.accepted[request_id] = state, []
            self.masks[request_id] = 0
            accept, mask = state.accept, state.mask

            def recorded(token):
                self.accepted[request_id].append(token)
                return accept(token)

            state.accept = recorded

            def counted(logits):
                self.masks[request_id] += 1
                mask(logits)

            state.mask = counted
        return request

    def _decode_batch(self, group, graph=None):
        self.batches.append(len(group))

        def checked(logits, requests):
            if graph is not None:
                assert torch.isfinite(logits).all(), "Graph输出残留上一轮原地掩码"
                self.fresh_logits += 1
            return sample_batch(logits, requests)

        with patch("ampere_kv.scheduler.sample_batch", side_effect=checked):
            super()._decode_batch(group, graph)

    def _retire(self, request):
        assert request.request_id not in self.retired
        self.retired.append(request.request_id)
        super()._retire(request)


def finish(scheduler, requests):
    for _ in range(512):
        if all(r.status == FINISHED for r in requests):
            break
        scheduler.step()
    else:
        raise AssertionError("结构化请求未在有限轮次结束")
    assert not scheduler.waiting and not scheduler.running and scheduler.prefilling is None and scheduler.reserved_blocks == 0
    assert all(s._pool.num_free_blocks == scheduler.blocks_per_layer for s in scheduler.storages)
    assert len(scheduler.retired) == len(requests)
    for request in requests:
        assert request.sampler.uniforms is request.sampler.structured is None
        if request.request_id in scheduler.states:
            state = scheduler.states[request.request_id]
            assert state.matcher is state.compiled is state.cpu_mask is state.device_mask is None
            assert scheduler.accepted[request.request_id] == request.output_ids
            assert scheduler.masks[request.request_id] == len(request.output_ids)


def check_model_path(model, tokenizer, path):
    stops = model.generation_config.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    compiler = SchemaCompiler(tokenizer, model.config.vocab_size, stops)
    category, approval = compiler.compile(CATEGORY), compiler.compile(APPROVAL)
    prompts = [encode_prompt(tokenizer, text) for text in (
        'Classify "prepare a meeting". Return only JSON with category work/personal/other and integer priority 1/2/3.',
        'Approve this harmless request. Return only a JSON object with boolean field approved.',
        'Say hello briefly.')]
    capacity = sum((p.shape[1] + n + 16) // 16 for p, n in
                   ((prompts[0], 128), (prompts[1], 64), (prompts[0], 128), (prompts[2], 6)))
    capacity = max(capacity, 4 * ((prompts[0].shape[1] + 128 + 16) // 16))
    width = ((max(p.shape[1] for p in prompts) + 128 + 15) // 16) * 16
    backend = "cuda" if path == "graph-cuda" else "reference"

    def make_scheduler(max_batch=4):
        scheduler = AuditedScheduler(model, capacity, max_batch=max_batch)
        if path != "eager-reference":
            scheduler.enable_graphs(width)
        return scheduler

    baseline = None
    for repeat in range(2):
        scheduler = make_scheduler()
        requests = [scheduler.submit("category", prompts[0], 128, structured=category),
                    scheduler.submit("approval", prompts[1], 64, structured=approval, temperature=0.8,
                                     top_k=40, seed=42, sampling_backend=backend),
                    scheduler.submit("category-random", prompts[0], 128, structured=category, temperature=0.8,
                                     top_k=128, seed=43, sampling_backend=backend),
                    scheduler.submit("plain", prompts[2], 6, ignore_eos=True)]
        assert scheduler.states["category"].matcher is not scheduler.states["approval"].matcher
        finish(scheduler, requests)
        for request, schema in zip(requests, (CATEGORY, APPROVAL, CATEGORY)):
            text = tokenizer.decode(request.output_ids, skip_special_tokens=True)
            assert request.structured_complete and request.stop_reason == "EOS"
            validate(text, schema)
            print(f"[JSON] 路径={path} 轮次={repeat} 请求={request.request_id} 输出={text} 停止={request.stop_reason} Token数={len(request.output_ids)}")
        assert len(requests[3].output_ids) == 6 and requests[3].stop_reason == "达到生成上限"
        output = [list(r.output_ids) for r in requests]
        if baseline is not None:
            assert output == baseline, "相同输入/成员/配置/seed复跑不同"
        baseline = output
        if path != "eager-reference":
            assert scheduler.graph_calls > 0 and scheduler.graph_fallbacks > 0
            assert scheduler.fresh_logits == scheduler.graph_calls
        print(f"[路径] {path} 批量={sorted(set(scheduler.batches))} Graph={scheduler.graph_calls} 回退={scheduler.graph_fallbacks}")
    print(f"[PASS] {path} 两Schema/普通请求、greedy/随机、掩码与提交各一次、完整JSON、同配置复现与回收")

    scheduler = make_scheduler(1)
    short = scheduler.submit("short", prompts[0], 1, structured=category)
    finish(scheduler, [short])
    assert not short.structured_complete and short.stop_reason == "达到生成上限（结构化截断）"
    print(f"[截断] 输出={tokenizer.decode(short.output_ids, skip_special_tokens=True)!r}，停止={short.stop_reason}，不补括号")

    scheduler = make_scheduler(1)
    neighbor = scheduler.submit("neighbor", prompts[2], 8, ignore_eos=True)
    active = scheduler.submit("active", prompts[0], 128, structured=category,
                              temperature=0.8, top_k=40, seed=42, sampling_backend=backend)
    waiting = scheduler.submit("waiting", prompts[0], 128, structured=category)
    expired = scheduler.submit("expired", prompts[0], 128, structured=category, deadline=time.monotonic() - 1)
    scheduler.cancel(waiting)
    scheduler.step(); scheduler.step()
    assert not waiting.output_ids and not expired.output_ids and len(active.output_ids) == 1
    owned = set(active.caches[0]._table.block_ids)
    scheduler.cancel(active)
    replacement = scheduler.submit("replacement", prompts[0], 128, structured=category,
                                   temperature=0.8, top_k=40, seed=42, sampling_backend=backend)
    scheduler.step()
    assert active.status == FINISHED and active.stop_reason == "已取消"
    assert set(replacement.caches[0]._table.block_ids) & owned
    replacement.deadline = time.monotonic() - 1
    scheduler.step()
    final = scheduler.submit("final", prompts[1], 64, structured=approval,
                             temperature=0.8, top_k=40, seed=42, sampling_backend=backend)
    finish(scheduler, [neighbor, active, waiting, expired, replacement, final])
    assert waiting.stop_reason == "已取消" and expired.stop_reason == replacement.stop_reason == "已超时"
    assert len(neighbor.output_ids) == 8 and neighbor.stop_reason == "达到生成上限"
    assert final.structured_complete
    validate(tokenizer.decode(final.output_ids, skip_special_tokens=True), APPROVAL)
    print(f"[PASS] {path} 等待/生成取消、等待/活动到期、替换复用、邻居完成及全部matcher/块回收")

    # 先分别Prefill，再同轮启动四条Decode，避免自然短输出在准入期间提前退场。
    scheduler = make_scheduler()
    requests = [scheduler.submit(f"shrink{i}", prompts[0], 128, structured=category,
                                 temperature=0.8, top_k=40, seed=42 + i, sampling_backend=backend)
                for i in range(4)]
    while scheduler.waiting:
        scheduler._admit()
        request = scheduler.prefilling
        scheduler._advance(request, prefill=True)
        assert not request.stop_reason
        scheduler.prefilling = None
        request.status = RUNNING
        scheduler.running.append(request)
    for index, expected in enumerate((4, 3, 2, 1)):
        if index:
            scheduler.cancel(requests[index - 1])
        _, sizes = scheduler.step()
        assert sizes == [expected], (path, sizes, expected)
    assert scheduler.batches == [4, 3, 2, 1]
    if path != "eager-reference":
        assert scheduler.graph_calls == 2 and scheduler.graph_fallbacks == 2
    finish(scheduler, requests)
    assert requests[-1].structured_complete
    validate(tokenizer.decode(requests[-1].output_ids, skip_special_tokens=True), CATEGORY)
    print(f"[PASS] {path} B4→B3→B2→B1；取消所属状态释放、邻居JSON完成，Graph按既有批次命中/回退")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--path", choices=(*PATHS, "all"), default="all")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    generation = GenerationConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    stops = generation.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    check_masks(tokenizer, config.vocab_size, stops)
    check_cuda_masks(tokenizer, config.vocab_size, stops)
    if args.model:
        model, tokenizer = load_model_and_tokenizer()
        for path in PATHS if args.path == "all" else (args.path,):
            check_model_path(model, tokenizer, path)
    print("[完成] 同步结构化输出检查")


if __name__ == "__main__":
    main()
