"""对称INT8量化参考；用于CPU对照和未融合的分页写入，CUDA单Token快路径另行实现。"""

import torch


@torch.no_grad()
def quantize_kv(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 BF16 [1, KV头, Token数, 维度]，返回 INT8 数据与 FP16 [..., 1] scale。

    K、V 分别调用，不共享 scale。清晰优先的参考实现，含 GPU 标量取回，不用于计时路径。
    """
    if not torch.isfinite(tensor).all().item():
        raise ValueError("量化输入不能含 NaN 或 Inf")
    values = tensor.float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    # 全零向量固定 scale=1；非零向量最小使用 FP16 最小正规数 2^-14，不依赖次正规数。
    candidate = (maximum / 127.0).clamp_min(torch.finfo(torch.float16).tiny)
    candidate = torch.where(maximum == 0, torch.ones_like(candidate), candidate)
    # 不默默截断过大的 scale：该参考明确拒绝 FP16 正常范围无法承载的输入。
    if (candidate > torch.finfo(torch.float16).max).any().item():
        raise ValueError("量化 scale 超过 FP16 有限范围")
    scale = candidate.to(torch.float16)
    # 用实际保存的 scale 量化；torch.round 半整数向最近偶数舍入，不用 -128 以保持对称。
    data = torch.round(values / scale.float()).clamp(-127, 127).to(torch.int8)
    return data, scale


@torch.no_grad()
def dequantize_kv(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """用保存的 FP16 scale 还原为 FP32 独立张量；仅作数值参考，不是融合路径。"""
    return data.float() * scale.float()
