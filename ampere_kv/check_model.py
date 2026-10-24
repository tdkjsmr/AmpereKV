"""模型诊断：CUDA/SDPA 生成对照与 INT8 同历史量化观测，不参与正常推理。"""

import statistics

import torch

from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.runner import (MAX_NEW_TOKENS, PAGED_BLOCK_SIZE, decoder_layer_forward,
                              final_logits, model_forward)


@torch.inference_mode()
def check_cuda_decode(model, input_ids, v3: bool = False) -> None:
    """两套独立分页缓存做有界贪心生成；首次选词不一致即停止，不强制喂参考 Token。

    v3 只作用于 CUDA 那一套；参考那一套始终走 SDPA，保持起点相同。
    """
    config = model.config
    if config.head_dim != 128 or PAGED_BLOCK_SIZE != 16:
        raise ValueError("CUDA Decode 只支持每头 128 维和块大小 16")
    tokens = input_ids.shape[1]
    num_blocks = (tokens + MAX_NEW_TOKENS + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    actual_ids, reference_ids = [], []
    # 两条路径共享只读权重，但不共享缓存张量或请求元数据。
    actual_caches, reference_caches = [], []
    try:
        for caches in (actual_caches, reference_caches):
            for _ in model.model.layers:
                caches.append(PagedKVCache(PagedKVStorage(
                    config.num_key_value_heads, config.head_dim, num_blocks,
                    PAGED_BLOCK_SIZE, device=input_ids.device,
                )))
        all_caches = actual_caches + reference_caches
        actual_input, reference_input = input_ids, input_ids
        print(f"对照上限={MAX_NEW_TOKENS} 个新 Token，输入长度={tokens}，每层物理容量={num_blocks * PAGED_BLOCK_SIZE}，CUDA 内核={'V3 四头共享载入' if v3 else 'V1 默认'}")
        for step in range(MAX_NEW_TOKENS):
            is_prefill = step == 0
            # 第一次 Decode 的位置为 tokens；此后每步只增加一个位置。
            positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0) if is_prefill else torch.tensor([[tokens + step - 1]], device=input_ids.device)
            # 各自完整走一遍模型，上一层自己的输出直接进入下一层，不能换回参考状态。
            actual = model_forward(model, actual_input, positions, actual_caches, is_prefill=is_prefill, cuda_decode=not is_prefill, v3=v3)
            reference = model_forward(model, reference_input, positions, reference_caches, is_prefill=is_prefill)
            assert torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item()
            used = tokens + step
            assert all(cache.length == used for cache in all_caches)
            actual_id, reference_id = actual.argmax(dim=-1).item(), reference.argmax(dim=-1).item()
            if is_prefill:
                # 起点必须相同，防止把 Prefill 的差异误归因于 CUDA Decode。
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                print(f"[PASS] 两套 SDPA Prefill logits 完全一致：首个 Token={actual_id}，KV 长度={used}")
            else:
                # BF16 logits 的差值转 FP32 统计；不直接套用 Attention 算子的容差。
                difference = actual.float() - reference.float()
                reference_norm = torch.linalg.vector_norm(reference.float()).item()
                relative_l2 = f"{torch.linalg.vector_norm(difference).item() / reference_norm:.8g}" if reference_norm != 0 else "N/A（参考全零）"
                print(f"[观测] Decode 第 {step} 步 logits：最大绝对误差={difference.abs().max().item():.8g}，相对 L2 误差={relative_l2}")
            actual_ids.append(actual_id)
            reference_ids.append(reference_id)
            if actual_id != reference_id:
                print(f"[分歧] 第 {step + 1} 个 Token：CUDA={actual_id}，SDPA={reference_id}，KV 长度={used}")
                # 只在分歧时补充候选，避免每步都打印大量词表诊断。
                # 打印 logits 分差；topk 的同分候选顺序由后端决定。
                for label, logits in (("CUDA", actual), ("SDPA", reference)):
                    scores, ids = logits[0, 0].float().topk(2)
                    print(f"[诊断] {label} 前两名 ID={ids.tolist()}，logits={scores.tolist()}，间隔={(scores[0] - scores[1]).item():.8g}")
                print(f"CUDA 序列={actual_ids}，SDPA 序列={reference_ids}")
                raise AssertionError("首次贪心选词不一致，停止后续生成；未将参考 Token 灌入 CUDA 路径")
            # 最终选出的 Token（包括 EOS）不再写入缓存，最终长度为 P + N - 1。
            if actual_id in eos_ids or step + 1 == MAX_NEW_TOKENS:
                break
            actual_input = torch.tensor([[actual_id]], dtype=torch.long, device=input_ids.device)
            reference_input = torch.tensor([[reference_id]], dtype=torch.long, device=input_ids.device)
        assert actual_ids == reference_ids
        crossed = (used - 1) // PAGED_BLOCK_SIZE > (tokens - 1) // PAGED_BLOCK_SIZE
        reason = "EOS" if actual_ids[-1] in eos_ids else "达到新 Token 上限"
        print(f"CUDA 序列={actual_ids}")
        print(f"SDPA 序列={reference_ids}")
        print(f"停止原因={reason}，生成数={len(actual_ids)}，Decode 次数={len(actual_ids) - 1}，最终 KV 长度={used}，Decode 跨块={crossed}")
        if len(actual_ids) == 1:
            print("[未覆盖] 首个 Token 为 EOS，本次未执行 CUDA Decode")
        elif not crossed:
            print("[未覆盖] 本次真实模型 Decode 没有跨块")
    finally:
        # 成功、断言失败或首个 Token 为 EOS，都归还已经建立的两套缓存。
        for cache in actual_caches + reference_caches:
            cache.release()
        print("已归还本次两套分页缓存")
    print("[PASS] 本次有界生成的 Token 序列与自建 SDPA 参考一致，缓存检查通过")


@torch.inference_mode()
def check_int8_decode(model, input_ids, v3: bool = False) -> None:
    """共享BF16 Prefill，按BF16选词驱动两条CUDA路径；仅同历史对照，不测性能。

    v3 同时作用于两套 CUDA 路径，使差异只来自精度而不是内核版本。
    """
    config = model.config
    length = input_ids.shape[1]
    blocks = (length + MAX_NEW_TOKENS - 1 + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    reference_caches, quantized_caches = [], []
    positions = torch.arange(length, device=input_ids.device)[None]
    hidden = model.model.embed_tokens.weight[input_ids]
    try:
        # 只走一次BF16 Prefill，逐层将同一份RoPE后的K和原始V保存成两种表示。
        # INT8缓存不参与Prefill Attention，避免起点已经是两套不同的隐藏状态。
        for layer in model.model.layers:
            for caches, dtype in ((reference_caches, torch.bfloat16), (quantized_caches, torch.int8)):
                storage = PagedKVStorage(config.num_key_value_heads, config.head_dim, blocks,
                                        PAGED_BLOCK_SIZE, device=input_ids.device, kv_dtype=dtype)
                caches.append(PagedKVCache(storage))
            hidden = decoder_layer_forward(hidden, layer, positions, reference_caches[-1], config,
                                           model._ampere_inv_freq, is_prefill=True)
            # get只在诊断准备阶段复制当前层历史；Decode直接由CUDA读取物理块。
            key, value = reference_caches[-1].get()
            quantized_caches[-1].append(key, value, fused=True)
        first_logits = final_logits(model, hidden)
        # 对照入口保留数值底线，避免NaN经argmax后被误报为选词一致。
        assert torch.isfinite(first_logits).all().item()
        first_token = first_logits.argmax(dim=-1)
        all_caches = reference_caches + quantized_caches
        assert all(cache.length == length for cache in all_caches)
        print(f"[PASS] 共用BF16 Prefill完成：层数={len(reference_caches)}，KV长度={length}，首Token={first_token.item()}")
        # 首Token来自共同Prefill，不计入Decode选词一致率；若它是EOS则零步结束。
        del hidden, first_logits, key, value
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        current_id = first_token.item()
        compared = matched = 0
        relative_errors = []
        kl_values, mismatch_gaps = [], []
        first_mismatch = None
        used = length
        for step in range(MAX_NEW_TOKENS - 1):
            if current_id in eos_ids:
                break
            # 两套缓存只共享Token历史，不共享中间隐藏状态或新计算的K/V。
            current_input = input_ids.new_tensor([[current_id]])
            positions = input_ids.new_tensor([[length + step]])
            reference = model_forward(model, current_input, positions, reference_caches,
                                      is_prefill=False, cuda_decode=True, v3=v3)
            actual = model_forward(model, current_input, positions, quantized_caches,
                                   is_prefill=False, cuda_decode=True, v3=v3)
            assert torch.isfinite(reference).all().item() and torch.isfinite(actual).all().item()
            reference_scores, actual_scores = reference.float().flatten(), actual.float().flatten()
            delta = actual_scores - reference_scores
            relative_errors.append((delta.norm() / reference_scores.norm().clamp_min(1e-12)).item())
            # 完整词表、温度1、自然对数：KL方向为BF16到INT8，不做Top-k截断。
            reference_logp = reference_scores.log_softmax(dim=-1)
            actual_logp = actual_scores.log_softmax(dim=-1)
            kl = (reference_logp.exp() * (reference_logp - actual_logp)).sum().item()
            kl_values.append(kl)  # 保留FP32原始结果；极小负值可能来自浮点舍入。
            reference_id = reference.argmax(dim=-1).item()
            actual_id = actual.argmax(dim=-1).item()
            compared += 1
            matched += int(reference_id == actual_id)
            if reference_id != actual_id:
                # 衡量实际竞争候选的差距，而不假设INT8选择的是BF16第二名。
                gap = (reference_scores[reference_id] - reference_scores[actual_id]).item()
                mismatch_gaps.append(gap)
                if first_mismatch is None:
                    top_two = reference_scores.topk(2).values
                    margin = (top_two[0] - top_two[1]).item()
                    # 并列使用竞争排名：严格更高的候选数+1，不对整个词表排序。
                    rank = (reference_scores > reference_scores[actual_id]).sum().item() + 1
                    ids = [reference_id, actual_id]
                    # 只保存CPU标量与两个候选分数，不保留逐步GPU logits。
                    first_mismatch = (step + 2, length + step, ids, kl, margin, rank,
                                      reference_scores[ids].tolist(), actual_scores[ids].tolist())
            # INT8即使选到EOS也不提前结束；下一轮始终服从BF16基线，避免文本历史分叉。
            current_id = reference_id
            used = length + compared
        reason = "BF16 EOS" if current_id in eos_ids else "达到输出上限"
        print(f"[汇总] 同历史对照：输出上限={MAX_NEW_TOKENS}，基线输出数={compared + 1}，Decode比较数={compared}，停止原因={reason}")
        if compared:
            print(f"[观测] Decode选词一致={matched}/{compared}（{matched / compared:.2%}），不含共用首Token")
            print(f"[观测] logits相对L2：均值={statistics.mean(relative_errors):.8g}，最大值={max(relative_errors):.8g}")
            print(f"[观测] KL(BF16 || INT8)：均值={statistics.mean(kl_values):.8g}，最大值={max(kl_values):.8g}（自然对数，温度1）")
            if mismatch_gaps:
                print(f"[观测] 分歧步BF16竞争分差：均值={statistics.mean(mismatch_gaps):.8g}，最大值={max(mismatch_gaps):.8g}，仅统计{len(mismatch_gaps)}个分歧步")
        else:
            print("[未覆盖] 首Token为EOS，没有执行Decode；一致率与误差无定义")
        if first_mismatch is not None:
            number, position, ids, kl, margin, rank, reference_pair, actual_pair = first_mismatch
            print(f"[观测] 首次分歧：第{number}个输出Token，输入位置={position}，BF16={ids[0]}，INT8={ids[1]}")
            print(f"[观测] 该步KL={kl:.8g}，BF16前两名分差={margin:.8g}，INT8所选Token在BF16中排名={rank}（严格更高数+1）")
            print(f"[观测] 候选顺序={ids}，BF16分数={reference_pair}，INT8分数={actual_pair}")
        elif compared:
            print("[观测] 本次Decode比较未出现选词分歧")
        crossed = compared > 0 and (used - 1) // PAGED_BLOCK_SIZE > (length - 1) // PAGED_BLOCK_SIZE
        assert all(cache.length == used for cache in all_caches)
        print(f"[PASS] 两套全部层KV长度={used}，Decode跨块={crossed}")
    finally:
        for cache in reference_caches + quantized_caches:
            cache.release()
    assert all(cache._pool.num_free_blocks == blocks for cache in reference_caches + quantized_caches)
    print("[PASS] BF16驱动的同历史对照完成，两套缓存块全部归还")
