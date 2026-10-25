"""结构化输出：默认检查真实tokenizer/掩码，--model运行同步模型与生命周期演示。"""

import argparse
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ampere_kv.runner import MODEL_ID, MODEL_REVISION, encode_prompt, load_model_and_tokenizer
from ampere_kv.sampler import RequestSampler, sample_batch
from ampere_kv.scheduler import FINISHED, Scheduler
from ampere_kv.structured_output import SchemaCompiler


CATEGORY = {"type": "object", "properties": {
    "category": {"type": "string", "enum": ["work", "personal", "other"]},
    "priority": {"type": "integer", "enum": [1, 2, 3]}},
    "required": ["category", "priority"], "additionalProperties": False}
APPROVAL = {"type": "object", "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"], "additionalProperties": False}


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
    for field, value in (("graphs", {}), ("decode_pipeline", object()), ("mixed", True), ("prefill_chunk_size", 16)):
        previous = getattr(scheduler, field)
        setattr(scheduler, field, value)
        rejected(lambda: scheduler.submit("bad", prompt, 16, structured=grammar))
        setattr(scheduler, field, previous)
    rejected(lambda: scheduler.submit("bad", prompt, 16, structured=grammar, ignore_eos=True))
    rejected(lambda: scheduler.submit("bad", prompt, 16, structured=grammar, sampling_backend="cuda"))
    assert not scheduler.waiting and not scheduler._submitted
    print("[PASS] 可选依赖不影响普通导入；结构化Graph/流水/mixed/分块/ignore_eos/CUDA采样在提交边界拒绝")


class AuditedScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states, self.accepted, self.retired = {}, {}, []

    def submit(self, request_id, *args, **kwargs):
        request = super().submit(request_id, *args, **kwargs)
        state = request.sampler.structured
        if state is not None:
            self.states[request_id], self.accepted[request_id] = state, []
            accept = state.accept

            def recorded(token):
                self.accepted[request_id].append(token)
                return accept(token)

            state.accept = recorded
        return request

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


def check_model():
    model, tokenizer = load_model_and_tokenizer()
    stops = model.generation_config.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    compiler = SchemaCompiler(tokenizer, model.config.vocab_size, stops)
    category, approval = compiler.compile(CATEGORY), compiler.compile(APPROVAL)
    prompts = [encode_prompt(tokenizer, text) for text in (
        'Classify "prepare a meeting". Return only JSON with category work/personal/other and integer priority 1/2/3.',
        'Approve this harmless request. Return only a JSON object with boolean field approved.',
        'Say hello briefly.')]
    capacity = sum((p.shape[1] + n + 16) // 16 for p, n in zip(prompts, (128, 64, 6)))
    baseline = None
    for repeat in range(2):
        scheduler = AuditedScheduler(model, capacity, max_batch=3)
        requests = [scheduler.submit("category", prompts[0], 128, structured=category),
                    scheduler.submit("approval", prompts[1], 64, structured=approval, temperature=0.8, top_k=40, seed=42),
                    scheduler.submit("plain", prompts[2], 6, ignore_eos=True)]
        assert scheduler.states["category"].matcher is not scheduler.states["approval"].matcher
        finish(scheduler, requests)
        for request, schema in zip(requests, (CATEGORY, APPROVAL)):
            text = tokenizer.decode(request.output_ids, skip_special_tokens=True)
            assert request.structured_complete and request.stop_reason == "EOS"
            validate(text, schema)
            print(f"[JSON] 轮次={repeat} 请求={request.request_id} 输出={text} 停止={request.stop_reason} Token数={len(request.output_ids)}")
        assert len(requests[2].output_ids) == 6 and requests[2].stop_reason == "达到生成上限"
        output = [list(r.output_ids) for r in requests]
        if baseline is not None:
            assert output == baseline, "相同输入/成员/配置/seed复跑不同"
        baseline = output
    print("[PASS] 两Schema与普通请求交错、greedy/固定seed有限K、逐提交所属matcher推进一次、同配置复现与回收")

    scheduler = AuditedScheduler(model, capacity)
    short = scheduler.submit("short", prompts[0], 1, structured=category)
    finish(scheduler, [short])
    assert not short.structured_complete and short.stop_reason == "达到生成上限（结构化截断）"
    print(f"[截断] 输出={tokenizer.decode(short.output_ids, skip_special_tokens=True)!r}，停止={short.stop_reason}，不补括号")

    scheduler = AuditedScheduler(model, capacity, max_batch=1)
    neighbor = scheduler.submit("neighbor", prompts[2], 8, ignore_eos=True)
    active = scheduler.submit("active", prompts[0], 128, structured=category)
    waiting = scheduler.submit("waiting", prompts[0], 128, structured=category)
    expired = scheduler.submit("expired", prompts[0], 128, structured=category, deadline=time.monotonic() - 1)
    scheduler.cancel(waiting)
    scheduler.step(); scheduler.step()
    assert not waiting.output_ids and not expired.output_ids and len(active.output_ids) == 1
    owned = set(active.caches[0]._table.block_ids)
    scheduler.cancel(active)
    replacement = scheduler.submit("replacement", prompts[0], 128, structured=category)
    scheduler.step()
    assert active.status == FINISHED and active.stop_reason == "已取消"
    assert set(replacement.caches[0]._table.block_ids) & owned
    replacement.deadline = time.monotonic() - 1
    scheduler.step()
    final = scheduler.submit("final", prompts[1], 64, structured=approval)
    finish(scheduler, [neighbor, active, waiting, expired, replacement, final])
    assert waiting.stop_reason == "已取消" and expired.stop_reason == replacement.stop_reason == "已超时"
    assert len(neighbor.output_ids) == 8 and neighbor.stop_reason == "达到生成上限"
    assert final.structured_complete
    validate(tokenizer.decode(final.output_ids, skip_special_tokens=True), APPROVAL)
    print("[PASS] 等待/生成取消、等待/活动到期、替换复用、邻居完成及全部matcher/块回收")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    generation = GenerationConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    stops = generation.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    check_masks(tokenizer, config.vocab_size, stops)
    if args.model:
        check_model()
    print("[完成] 同步结构化输出检查")


if __name__ == "__main__":
    main()
