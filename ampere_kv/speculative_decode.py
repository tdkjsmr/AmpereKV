"""单请求BF16 greedy投机解码：独立草稿、目标验证与分页尾部截断。"""

import argparse

import torch
from transformers import AutoTokenizer

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (MODEL_ID, MODEL_REVISION, encode_prompt,
                              load_model_and_tokenizer, model_forward)

DRAFT_ID = "Qwen/Qwen3-0.6B"
DRAFT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def eos_ids(model):
    value = model.generation_config.eos_token_id
    return {value} if isinstance(value, int) else set(value or [])


def load_pair():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_ID, revision=DRAFT_REVISION, local_files_only=True)
    if (tokenizer.get_vocab() != draft_tokenizer.get_vocab()
            or tokenizer.special_tokens_map != draft_tokenizer.special_tokens_map
            or tokenizer.all_special_ids != draft_tokenizer.all_special_ids):
        raise ValueError("目标与草稿的词表ID映射或特殊Token不一致")
    target, _ = load_model_and_tokenizer(tokenizer=tokenizer)
    draft, _ = load_model_and_tokenizer(DRAFT_ID, DRAFT_REVISION, tokenizer=draft_tokenizer)
    if target.config.vocab_size != draft.config.vocab_size or eos_ids(target) != eos_ids(draft):
        raise ValueError("目标与草稿的模型词表大小或EOS配置不一致")
    if draft.config.tie_word_embeddings and draft.lm_head.weight.data_ptr() != draft.model.embed_tokens.weight.data_ptr():
        raise ValueError("草稿模型的Embedding与LM Head未按配置绑定")
    print(f"[模型] 目标={MODEL_ID}@{MODEL_REVISION}，草稿={DRAFT_ID}@{DRAFT_REVISION}")
    print(f"[配置] 草稿层={draft.config.num_hidden_layers}，Q/KV头="
          f"{draft.config.num_attention_heads}/{draft.config.num_key_value_heads}；词表及特殊Token一致，使用目标输入模板")
    return target, draft, tokenizer


def make_caches(model, capacity):
    config = model.config
    return [PagedKVCache(PagedKVStorage(config.num_key_value_heads, config.head_dim,
                                       (capacity + 15) // 16, 16, device=model.device))
            for _ in model.model.layers]


def forward_tokens(model, ids, caches, *, all_logits=False, cuda_decode=False):
    start = caches[0].length
    positions = torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)
    return model_forward(model, ids, positions, caches, is_prefill=start == 0 or ids.shape[1] > 1,
                         all_logits=all_logits, cuda_decode=cuda_decode and start > 0 and ids.shape[1] == 1)


def select_prefix(proposals, predictions, remaining, stops, ignore_eos=False):
    """行i预测草稿i，末行预测额外Token；接受数只计真正提交的草稿。"""
    if len(predictions) != len(proposals) + 1 or remaining < 1:
        raise ValueError("验证须有K+1行预测且至少剩余一个输出位置")
    matched = next((i for i, (draft, target) in enumerate(zip(proposals, predictions))
                    if draft != target), len(proposals))
    committed = (proposals[:matched] + [predictions[matched]])[:remaining]
    if not ignore_eos:
        end = next((i + 1 for i, token in enumerate(committed) if token in stops), len(committed))
        committed = committed[:end]
    return committed, min(matched, len(committed)), matched


@torch.inference_mode()
def generate_speculative(target, draft, input_ids, *, draft_tokens=2, max_new_tokens=64,
                         ignore_eos=False, observer=None, diagnostic=None, draft_backend="reference",
                         draft_graph=None):
    """轮首两套KV均不含最后一个已输出Token；observer仅供正确性检查读取。"""
    if draft_tokens not in (2, 4) or not 1 <= max_new_tokens <= 64:
        raise ValueError("草稿长度仅支持2/4，输出上限须为1至64")
    if draft_backend not in ("reference", "cuda", "graph"):
        raise ValueError("草稿后端须为reference/cuda/graph")
    if draft_backend == "graph" and (draft_graph is None or draft_graph.model is not draft):
        raise ValueError("graph后端须提供该草稿模型的常驻图")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("输入须为单条非空Token序列")
    if input_ids.dtype != torch.long or not input_ids.is_cuda:
        raise ValueError("输入须为CUDA int64 Token IDs")
    target_caches, draft_caches = [], []
    stops = eos_ids(target)
    rounds, proposed, accepted, draft_forwards = [], 0, 0, 1
    graph_before = draft_graph.counts() if draft_backend == "graph" else None
    try:
        if diagnostic is not None:
            diagnostic.start("allocate")
        target_caches = make_caches(target, input_ids.shape[1] + max_new_tokens)
        draft_caches = (draft_graph.acquire(input_ids.shape[1] + max_new_tokens) if graph_before is not None
                        else make_caches(draft, input_ids.shape[1] + max_new_tokens))
        if diagnostic is not None:
            diagnostic.end()
            diagnostic.start("target_prefill", gpu=True)
        first_logits = forward_tokens(target, input_ids, target_caches)
        if diagnostic is not None:
            diagnostic.end()
            diagnostic.start("draft_prefill", gpu=True)
        forward_tokens(draft, input_ids, draft_caches)
        if diagnostic is not None:
            diagnostic.end()
            diagnostic.start("first_selection")
        outputs = [first_logits[0, -1].argmax().item()]
        if diagnostic is not None:
            diagnostic.end()
        if observer is not None:
            observer(dict(prefill=True, output_start=0, committed=outputs.copy(), length=input_ids.shape[1]),
                     target_caches, draft_caches, first_logits)
        while len(outputs) < max_new_tokens and (ignore_eos or outputs[-1] not in stops):
            old_length = target_caches[0].length
            remaining = max_new_tokens - len(outputs)
            count = min(draft_tokens, remaining - 1)
            pending, proposals = outputs[-1], []
            token = pending
            if diagnostic is not None:
                diagnostic.start("drafting", gpu=True)
            if graph_before is not None:
                round_device = draft_graph.propose(pending, count)
            else:
                for _ in range(count):
                    logits = forward_tokens(draft, input_ids.new_tensor([[token]]), draft_caches,
                                            cuda_decode=draft_backend == "cuda")
                    token = logits[0, -1].argmax().item()
                    proposals.append(token)
            if diagnostic is not None:
                diagnostic.end()
                diagnostic.start("verification", gpu=True)
            if graph_before is not None:
                logits = forward_tokens(target, round_device, target_caches, all_logits=True)
                # 候选与目标预测只在本轮验证边界一起取回，不在K个草稿步之间取词。
                values = torch.cat((round_device[0], logits[0].argmax(-1))).tolist()
                round_input, predictions = values[:count + 1], values[count + 1:]
                proposals = round_input[1:]
            else:
                round_input = [pending, *proposals]
                logits = forward_tokens(target, input_ids.new_tensor([round_input]), target_caches, all_logits=True)
                predictions = logits[0].argmax(-1).tolist()
            if diagnostic is not None:
                diagnostic.end()
                diagnostic.start("tail_state")
            committed, kept, matched = select_prefix(proposals, predictions, remaining, stops, ignore_eos)
            length = old_length + len(committed)
            for cache in target_caches:
                cache.truncate(length)
            for cache in draft_caches:
                cache.truncate(min(length, cache.length))
            # 全接受时草稿尚未输入最后一个候选；补算真实KV，不能只增加逻辑长度。
            missing = round_input[draft_caches[0].length - old_length:length - old_length]
            if diagnostic is not None:
                diagnostic.end()
            if missing:
                if diagnostic is not None:
                    diagnostic.start("repair", gpu=True)
                if graph_before is not None:
                    begin = draft_caches[0].length - old_length
                    draft_graph.repair(round_device[:, begin:begin + len(missing)])
                else:
                    forward_tokens(draft, input_ids.new_tensor([missing]), draft_caches, cuda_decode=draft_backend == "cuda")
                if diagnostic is not None:
                    diagnostic.end()
            if diagnostic is not None:
                diagnostic.start("commit_state")
            draft_forwards += count + bool(missing)
            event = dict(prefill=False, output_start=len(outputs), old_length=old_length,
                         input_ids=round_input, proposals=proposals, predictions=predictions,
                         matched=matched, accepted=kept, committed=committed, length=length)
            outputs.extend(committed)
            proposed += count
            accepted += kept
            rounds.append(event)
            if observer is not None:
                observer(event, target_caches, draft_caches, logits)
            if diagnostic is not None:
                diagnostic.end()
        result = dict(output_ids=outputs, stop_reason="EOS" if not ignore_eos and outputs[-1] in stops else "达到生成上限",
                      proposed=proposed, accepted=accepted, verification_rounds=len(rounds),
                      target_forwards=1 + len(rounds), draft_forwards=draft_forwards,
                      final_kv_length=target_caches[0].length, rounds=rounds, draft_backend=draft_backend,
                      final_kv_lengths={"target": [c.length for c in target_caches],
                                        "draft": [c.length for c in draft_caches]})
    finally:
        if diagnostic is not None:
            diagnostic.start("release")
        for cache in target_caches + ([] if graph_before is not None else draft_caches):
            cache.release()
        if graph_before is not None and draft_caches:
            draft_graph.release()
        if diagnostic is not None:
            diagnostic.end()
    result["recovered"] = all(cache.length == 0 and cache._pool.num_free_blocks == cache._key.shape[0]
                              for cache in target_caches + draft_caches)
    if graph_before is not None:
        after = draft_graph.counts()
        result["draft_graph"] = dict(captures=after["captures"], capture_delta=after["captures"] - graph_before["captures"],
            **{key: after[key] - graph_before[key] for key in ("replays", "device_tokens", "repairs")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="hi")
    parser.add_argument("--draft-tokens", type=int, choices=(2, 4), default=2)
    parser.add_argument("--draft-backend", choices=("reference", "cuda", "graph"), default="reference")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if not args.prompt.strip() or not 1 <= args.max_new_tokens <= 64:
        parser.error("输入不能为空，输出上限须为1至64")
    target, draft, tokenizer = load_pair()
    ids = encode_prompt(tokenizer, args.prompt)
    graph = None
    if args.draft_backend == "graph":
        from ampere_kv.speculative_draft import DraftGraph
        graph = DraftGraph(draft, ids.shape[1] + args.max_new_tokens)
    result = generate_speculative(target, draft, ids, draft_tokens=args.draft_tokens,
                                  max_new_tokens=args.max_new_tokens, ignore_eos=args.ignore_eos,
                                  draft_backend=args.draft_backend, draft_graph=graph)
    print(f"generated_token_ids = {result['output_ids']}")
    print(f"generated_text = {tokenizer.decode(result['output_ids'], skip_special_tokens=True)!r}")
    print(f"[统计] 草稿={result['draft_backend']}，输入={ids.shape[1]}，K={args.draft_tokens}，输出={len(result['output_ids'])}，"
          f"提议={result['proposed']}，接受={result['accepted']}，验证轮={result['verification_rounds']}，"
          f"接受长度={[r['accepted'] for r in result['rounds']]}，停止={result['stop_reason']}，回收={result['recovered']}")


if __name__ == "__main__":
    main()
