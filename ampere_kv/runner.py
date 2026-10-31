"""自建 Qwen3 模型前向、单请求生成与可选 HF 对照。"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ampere_kv.kv_cache import ContiguousKVCache, prefill_attention, decode_attention, sdpa_attention
from ampere_kv.paged_cache import PagedKVCache, PagedKVStorage
from ampere_kv.sampler import RequestSampler
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

MAX_NEW_TOKENS = 32
PAGED_BLOCK_SIZE = 16


def encode_prompt(tokenizer, text: str) -> torch.Tensor:
    """按 Qwen3 对话模板把一句输入编码成 [1, P] 的 CUDA Token IDs。"""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(formatted, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")


def load_model_and_tokenizer(model_id=MODEL_ID, revision=MODEL_REVISION, *, tokenizer=None):
    """加载指定固定版本；默认Qwen3-8B，模型独立持有RoPE频率。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    model.eval()
    if model.config.hidden_act != "silu" or any(layer.self_attn.sliding_window is not None for layer in model.model.layers):
        raise ValueError("当前只支持SiLU且无滑动窗口的Qwen3")
    dim = model.config.head_dim
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
    inv_freq = (1.0 / (model.config.rope_theta ** exponent)).to(model.model.embed_tokens.weight.device)
    model.register_buffer("_ampere_inv_freq", inv_freq, persistent=False)
    return model, tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True)


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """沿最后一维做 RMSNorm；按 Qwen3 的顺序保留归一化后的类型转换位置。"""

    # BF16 输入先转 FP32，降低平方和归约过程中的舍入误差。
    compute_states = hidden_states.float()
    mean_square = compute_states.square().mean(dim=-1, keepdim=True)
    normalized = compute_states * torch.rsqrt(mean_square + eps)
    # 先转回输入类型，再乘真实缩放权重；不能擅自换成乘完权重后才转 BF16。
    return normalized.to(hidden_states.dtype) * weight


def project_qkv(normalized: torch.Tensor, attention) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用真实权重投影并拆头，完成 Q/K 每头归一化；尚未应用 RoPE。"""

    query = torch.nn.functional.linear(normalized, attention.q_proj.weight, attention.q_proj.bias)
    key = torch.nn.functional.linear(normalized, attention.k_proj.weight, attention.k_proj.bias)
    value = torch.nn.functional.linear(normalized, attention.v_proj.weight, attention.v_proj.bias)
    head_shape = (*normalized.shape[:-1], -1, attention.head_dim)
    query = query.reshape(head_shape)
    key = key.reshape(head_shape)
    value = value.reshape(head_shape)
    query = rms_norm(query, attention.q_norm.weight, attention.q_norm.variance_epsilon)
    key = rms_norm(key, attention.k_norm.weight, attention.k_norm.variance_epsilon)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def apply_rope(query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor, config, inv_freq: torch.Tensor):
    """固定Qwen3默认RoPE，按前后两半配对旋转Q/K；固定频率由模型加载时创建。"""
    dim = config.head_dim
    with torch.autocast(device_type=query.device.type, enabled=False):
        frequencies = inv_freq[None, :, None].expand(query.shape[0], -1, 1)
        angles = (frequencies @ position_ids[:, None, :].to(device=query.device, dtype=torch.float32)).transpose(1, 2)
        # Qwen3 将前后两半配对，不是相邻偶/奇维度配对，因此角度表复制成两半。
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(query.dtype).unsqueeze(1)
        sin = angles.sin().to(query.dtype).unsqueeze(1)
    half = dim // 2
    rotated_query = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
    rotated_key = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
    # 保持 HF 的逐项乘法再相加顺序，旋转后仍是 BF16；这里不涉及 Attention。
    return query * cos + rotated_query * sin, key * cos + rotated_key * sin


@torch.no_grad()
def decoder_layer_forward(
    hidden_states: torch.Tensor, layer, position_ids: torch.Tensor,
    cache: ContiguousKVCache | PagedKVCache, config, inv_freq: torch.Tensor,
    *, is_prefill: bool, cuda_decode: bool = False,
    v3: bool = False,
) -> torch.Tensor:
    """单请求层前向：Prefill用SDPA，Decode可选CUDA/V3；写入旋转后的K与原V。"""
    attention = layer.self_attn
    norm = layer.input_layernorm
    normalized = rms_norm(hidden_states, norm.weight, norm.variance_epsilon)
    query, key, value = project_qkv(normalized, attention)
    query, key = apply_rope(query, key, position_ids, config, inv_freq)
    fused_prefill = (is_prefill and isinstance(cache, PagedKVCache) and cache._key.is_cuda
                     and cache._key.shape[2] == 16 and cache._key.shape[3] == 128)
    if fused_prefill:
        history = cache.length
        cache.append(key, value, fused=True)
        # 首块沿用原始 BF16 K/V；后续块读回包含本块的完整历史。读回成本属于分块参考路径。
        if history:
            key, value = cache.get()
        head_output = sdpa_attention(query, key, value, history=history, causal=True)
    elif cuda_decode:
        from ampere_kv import _C

        table = cache.append(key, value, fused=True)
        # 块号越界改由内核在访存前用设备端断言拦截：属异步失败，不可捕获后复用同一
        # CUDA 上下文，因此内部路径出现非法块表视为实现错误，本次运行终止。
        if cache._key.dtype == torch.int8:
            head_output = _C.paged_decode_int8_internal(
                query.contiguous(), cache._key, cache._value,
                cache._key_scale, cache._value_scale, table, cache.length, v3=v3,
            )
        else:
            head_output = _C.paged_decode_internal(query.contiguous(), cache._key, cache._value, table, cache.length, v3=v3)
    else:
        attention_fn = prefill_attention if is_prefill else decode_attention
        head_output = attention_fn(query, key, value, cache)
    return finish_decoder_layer(hidden_states, head_output, layer)


def finish_decoder_layer(hidden_states: torch.Tensor, head_output: torch.Tensor, layer):
    """输出投影、两次残差与MLP；残差使用未归一化的原输入。"""
    attention = layer.self_attn
    batch, tokens, _ = hidden_states.shape
    merged = head_output.transpose(1, 2).contiguous().reshape(batch, tokens, -1)
    attention_output = torch.nn.functional.linear(merged, attention.o_proj.weight, attention.o_proj.bias)
    after_attention = hidden_states + attention_output

    mlp_norm = layer.post_attention_layernorm
    mlp_input = rms_norm(after_attention, mlp_norm.weight, mlp_norm.variance_epsilon)
    mlp = layer.mlp
    gate = torch.nn.functional.linear(mlp_input, mlp.gate_proj.weight, mlp.gate_proj.bias)
    up = torch.nn.functional.linear(mlp_input, mlp.up_proj.weight, mlp.up_proj.bias)
    gated = torch.nn.functional.silu(gate) * up
    mlp_output = torch.nn.functional.linear(gated, mlp.down_proj.weight, mlp.down_proj.bias)
    return after_attention + mlp_output


@torch.no_grad()
def decoder_layer_batched_forward(
    hidden_states: torch.Tensor, layer, positions: torch.Tensor, caches: list, config, inv_freq: torch.Tensor,
    metadata=None,
) -> torch.Tensor:
    """每请求一个Token；本层缓存共享物理池，各自保留位置、块表和长度。"""
    attention = layer.self_attn
    normalized = rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    query, key, value = project_qkv(normalized, attention)
    query, key = apply_rope(query, key, positions, config, inv_freq)
    head_output = batched_decode_attention(query, key, value, caches, metadata)
    return finish_decoder_layer(hidden_states, head_output, layer)


def prepare_batched_decode(caches, device, width=None):
    """在宿主端一次登记本层新 Token，并准备写入起点与设备块表。"""
    if width is not None and any(cache.length >= width * cache._key.shape[2] for cache in caches):
        raise ValueError("固定块表容量不足")
    # 起点必须在登记新块之前取：reserve 之后 length 已含本步 Token，而写入位置仍是旧长度。
    starts = torch.tensor([cache.length for cache in caches], dtype=torch.long, device=device)
    rows = [cache.reserve(1) for cache in caches]
    width = max(len(row) for row in rows) if width is None else width
    # 块表按行定宽填充 0；内核只索引到各请求自己的有效长度，填充列不会被读到，
    # 越界情况由内核在读表之前的设备端断言拦截。
    table = torch.tensor([list(row) + [0] * (width - len(row)) for row in rows],
                         dtype=torch.long, device=device)
    return starts, table


def batched_decode_attention(query, key, value, caches, metadata=None):
    """复用批量写入与 V1 Attention；预备模式不执行宿主端预留。"""
    from ampere_kv import _C

    shared_key, shared_value = caches[0]._key, caches[0]._value
    starts, table = metadata if metadata is not None else prepare_batched_decode(caches, query.device)
    _C.bf16_write_batched(key.contiguous(), value.contiguous(), shared_key, shared_value, table, starts)
    return _C.paged_decode_batched(query.contiguous(), shared_key, shared_value, table, starts)


@torch.no_grad()
def model_forward_mixed(model, decode_ids, decode_positions, layer_caches,
                        prefill_ids, prefill_positions, prefill_caches, *, output_prefill_logits=False):
    """打包B个Decode和一个Prefill块，共用投影/MLP；Attention与KV仍按请求独立。"""
    batch = decode_ids.shape[0]
    ids = torch.cat((decode_ids.reshape(1, batch), prefill_ids), dim=1)
    positions = torch.cat((decode_positions.reshape(1, batch), prefill_positions), dim=1)
    hidden = model.model.embed_tokens.weight[ids]
    inv_freq = model._ampere_inv_freq
    for layer, caches, prefill_cache in zip(model.model.layers, layer_caches, prefill_caches):
        norm = layer.input_layernorm
        normalized = rms_norm(hidden, norm.weight, norm.variance_epsilon)
        query, key, value = project_qkv(normalized, layer.self_attn)
        query, key = apply_rope(query, key, positions, model.config, inv_freq)
        dq, dk, dv = (tensor[:, :, :batch].permute(2, 1, 0, 3) for tensor in (query, key, value))
        decoded = batched_decode_attention(dq, dk, dv, caches).permute(2, 1, 0, 3)
        pq, pk, pv = (tensor[:, :, batch:] for tensor in (query, key, value))
        history = prefill_cache.length
        prefill_cache.append(pk, pv, fused=True)
        if history:
            pk, pv = prefill_cache.get()
        prefilled = sdpa_attention(pq, pk, pv, history=history, causal=True)
        hidden = finish_decoder_layer(hidden, torch.cat((decoded, prefilled), dim=2), layer)
    selected = hidden[:, :batch]
    if output_prefill_logits:
        selected = torch.cat((selected, hidden[:, -1:]), dim=1)
    logits = final_logits(model, selected.transpose(0, 1))
    return logits[:batch], logits[batch:] if output_prefill_logits else None


@torch.no_grad()
def model_forward_batched(model, token_ids: torch.Tensor, positions: torch.Tensor, layer_caches: list,
                          metadata=None):
    """批量Decode返回[B,1,词表]；layer_caches按层/请求排列，Prefill走model_forward。"""
    hidden_states = model.model.embed_tokens.weight[token_ids]
    inv_freq = model._ampere_inv_freq
    for index, (layer, caches) in enumerate(zip(model.model.layers, layer_caches)):
        prepared = None if metadata is None else metadata[index]
        hidden_states = decoder_layer_batched_forward(hidden_states, layer, positions, caches,
                                                       model.config, inv_freq, prepared)
    return final_logits(model, hidden_states)


@torch.no_grad()
def model_forward(model, input_ids, position_ids, caches, *, is_prefill: bool, cuda_decode: bool = False,
                  v3: bool = False, output_logits: bool = True, all_logits: bool = False):
    """自建前向；默认返回末位置，all_logits显式返回本次所有输入位置的logits。"""
    if is_prefill and caches[0].length and caches[0]._key.dtype == torch.int8:
        raise ValueError("分块 Prefill 当前只支持 BF16 KV")
    layers = model.model.layers
    hidden_states = model.model.embed_tokens.weight[input_ids]
    inv_freq = model._ampere_inv_freq
    for layer, cache in zip(layers, caches):
        hidden_states = decoder_layer_forward(
            hidden_states, layer, position_ids, cache, model.config, inv_freq,
            is_prefill=is_prefill, cuda_decode=cuda_decode, v3=v3,
        )
    return final_logits(model, hidden_states, all_positions=all_logits) if output_logits else None


def final_logits(model, hidden_states, *, all_positions=False):
    """末尾归一化和LM Head；默认只返回末位置，不读写缓存。"""
    norm = model.model.norm
    normalized = rms_norm(hidden_states, norm.weight, norm.variance_epsilon)
    return torch.nn.functional.linear(normalized if all_positions else normalized[:, -1:, :],
                                      model.lm_head.weight, model.lm_head.bias)


@torch.inference_mode()
def generate_tokens(model, input_ids, *, max_new_tokens: int = MAX_NEW_TOKENS,
                    verify: bool = False, cache_kind: str = "contiguous", cuda_decode: bool = False,
                    kv_dtype: torch.dtype = torch.bfloat16, ignore_eos: bool = False, v3: bool = False,
                    temperature: float = 0.0, top_k: int = -1, seed: int | None = None,
                    sampling_backend: str = "reference") -> list[int]:
    """单请求生成并释放缓存；verify执行HF对照，cuda_decode仅切换Decode路径。"""
    if temperature != 0 and (verify or kv_dtype != torch.bfloat16 or v3):
        raise ValueError("随机生成仅覆盖BF16/V1非对照路径")
    sampler = RequestSampler(max_new_tokens, model.config.vocab_size, input_ids.device,
                             temperature=temperature, top_k=top_k, seed=seed, sampling_backend=sampling_backend)
    if temperature > 0:
        print(f"[采样] temperature={temperature}，top_k={top_k}，seed={sampler.seed}")
    tokens = input_ids.shape[1]
    capacity = tokens + max_new_tokens
    caches = []
    for _ in model.model.layers:
        if cache_kind == "paged":
            cache = PagedKVCache(PagedKVStorage(
                model.config.num_key_value_heads, model.config.head_dim,
                num_blocks=(capacity + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE,
                block_size=PAGED_BLOCK_SIZE, device=input_ids.device, kv_dtype=kv_dtype,
            ))
        else:
            cache = ContiguousKVCache(
                model.config.num_key_value_heads, model.config.head_dim,
                capacity=capacity, device=input_ids.device,
            )
        caches.append(cache)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    generated_ids = []
    used_tokens = 0
    current_input = input_ids
    if verify:
        reference_input, reference_cache, reference_ids = input_ids, None, []
        print(f"缓存={cache_kind}，预留长度={capacity}，每层物理容量={caches[0].capacity}")

    for step in range(max_new_tokens):
        positions = torch.arange(used_tokens, used_tokens + current_input.shape[1], device=input_ids.device).unsqueeze(0)
        logits = model_forward(model, current_input, positions, caches, is_prefill=(step == 0), cuda_decode=cuda_decode and step > 0, v3=v3)
        used_tokens += current_input.shape[1]
        next_id = sampler.sample(logits[0, 0], step).item()
        generated_ids.append(next_id)

        if verify:
            reference = model(
                input_ids=reference_input, position_ids=positions, cache_position=positions[0],
                past_key_values=reference_cache, use_cache=True, logits_to_keep=1, return_dict=True,
            )
            reference_cache = reference.past_key_values
            torch.testing.assert_close(logits, reference.logits, rtol=0, atol=0)
            reference_ids.append(reference.logits[0, 0].argmax().item())
            assert next_id == reference_ids[-1]
            for index, cache in enumerate(caches):
                assert cache.length == reference_cache.get_seq_length(index) == used_tokens

        # 先判断停止，再准备下一次输入；最终 Token（包括 EOS）不再写入 KV。
        if (not ignore_eos and next_id in eos_ids) or step + 1 == max_new_tokens:
            break
        current_input = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
        if verify:
            reference_input = torch.tensor([[reference_ids[-1]]], dtype=torch.long, device=input_ids.device)

    if verify:
        print(f"reference_token_ids = {reference_ids}")
        print("[PASS] 逐步 logits、完整 Token 序列与缓存检查通过")
        if cache_kind == "paged":
            crossed = (used_tokens - 1) // PAGED_BLOCK_SIZE > (tokens - 1) // PAGED_BLOCK_SIZE
            print(f"分页块大小={PAGED_BLOCK_SIZE}，Prefill 长度={tokens}，最终 KV 长度={used_tokens}，Decode 跨块={crossed}")
            if not crossed:
                print("[未覆盖] 本次 Decode 未跨块，需换输入补验；不能据此宣称跨块生成验证通过")
    if cache_kind == "paged":
        for cache in caches:
            cache.release()
        if verify:
            print("已归还全部层分页块；本入口不再重复检查块池内部标记")
    sampler.release()
    return generated_ids


def main() -> None:
    """加载与分词只做一次；选择对照或纯生成模式，最后统一展示结果。"""
    parser = argparse.ArgumentParser(description="Qwen3 自建 BF16 生成与正确性对照")
    parser.add_argument("--mode", choices=("check", "generate", "cuda-check", "int8-check"), default="check", help="check：HF 对照；generate：纯生成；cuda-check：有界 CUDA Decode 对照；int8-check：BF16驱动的INT8同历史对照")
    parser.add_argument("--cache", choices=("contiguous", "paged"), default="contiguous", help="默认连续缓存；paged 为每块 16 Token 的分页参考")
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="bf16",
                        help="生成的KV类型；int8需paged；诊断模式自行选择")
    parser.add_argument("--decode-kernel", choices=("v1", "v3"), default="v1",
                        help="Decode CUDA 内核：v1 每块一个 Query 头（默认），v3 同组四个 Query 头共享一次 K/V 载入")
    parser.add_argument("--temperature", type=float, default=0.0, help="0保留greedy；正数启用请求级采样")
    parser.add_argument("--top-k", type=int, default=-1, help="-1为全词表，或1..vocab_size")
    parser.add_argument("--seed", type=int, default=None, help="请求种子0..2^32-1；随机模式省略则选择并打印种子")
    parser.add_argument("--sampling-backend", choices=("reference", "cuda"), default="reference")
    args = parser.parse_args()
    if (args.temperature != 0 or args.top_k != -1 or args.seed is not None or args.sampling_backend != "reference") and (
            args.mode != "generate" or args.kv_dtype != "bf16" or args.decode_kernel != "v1"):
        parser.error("采样参数仅用于generate的BF16/V1路径")
    if args.mode in ("cuda-check", "int8-check") and args.cache != "paged":
        parser.error("CUDA 对照模式需要显式指定 --cache paged")
    if args.kv_dtype == "int8" and (args.mode != "generate" or args.cache != "paged"):
        parser.error("--kv-dtype int8 仅用于 generate 且需 --cache paged")
    if args.decode_kernel == "v3" and not (
            args.mode in ("cuda-check", "int8-check")
            or (args.mode == "generate" and args.kv_dtype == "int8")):
        parser.error("--decode-kernel v3 需要实际走 CUDA Decode 的组合：cuda-check/int8-check 或 generate --kv-dtype int8")
    if not torch.cuda.is_available():
        raise RuntimeError("模型生成需要在云端 CUDA 环境运行")
    text = input("请输入一段文本：")
    if not text.strip():
        raise ValueError("输入不能为空")
    print(f"正在从本地缓存加载固定版本 Qwen3-8B，模式={args.mode}，缓存={args.cache}……")
    model, tokenizer = load_model_and_tokenizer()
    input_ids = encode_prompt(tokenizer, text)
    if args.mode == "int8-check":
        from ampere_kv.check_model import check_int8_decode

        check_int8_decode(model, input_ids, v3=args.decode_kernel == "v3")
        return
    if args.mode == "cuda-check":
        from ampere_kv.check_model import check_cuda_decode

        check_cuda_decode(model, input_ids, v3=args.decode_kernel == "v3")
        return
    use_int8 = args.kv_dtype == "int8"
    if args.mode == "generate":
        print(f"生成路径：KV={args.kv_dtype}，Prefill=BF16 SDPA，Decode={'INT8 CUDA' if use_int8 else 'BF16 SDPA'}，Decode内核={args.decode_kernel}")
    generated_ids = generate_tokens(
        model, input_ids, verify=(args.mode == "check"), cache_kind=args.cache,
        cuda_decode=use_int8, kv_dtype=torch.int8 if use_int8 else torch.bfloat16,
        v3=args.decode_kernel == "v3",
        temperature=args.temperature, top_k=args.top_k, seed=args.seed, sampling_backend=args.sampling_backend,
    )
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    reason = "EOS" if generated_ids[-1] in eos_ids else "达到新 Token 上限"
    print(f"generated_token_ids = {generated_ids}")
    print(f"generated_text = {tokenizer.decode(generated_ids, skip_special_tokens=True)!r}")
    print(f"停止原因={reason}，生成数={len(generated_ids)}，Decode 次数={len(generated_ids) - 1}")
    if len(generated_ids) == 1:
        print("首个 Token 为 EOS，本次未覆盖 Decode")
    if args.mode == "generate":
        print("生成完成")


if __name__ == "__main__":
    main()
