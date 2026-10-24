"""请求级随机流与模型 Graph 外的 GPU 选词。"""

import math
import secrets

import torch


def distribution(logits, temperature, top_k):
    """候选按分数降序、同分Token ID升序；选足k项后才归一化。"""
    scores, ids = torch.sort(logits.float(), descending=True, stable=True)
    if top_k != -1:
        scores, ids = scores[:top_k], ids[:top_k]
    probabilities = torch.softmax((scores - scores[0]) / temperature, dim=-1)
    return ids, probabilities


def sample_uniform(logits, temperature, top_k, uniform):
    """给定[0,1)的uniform，取首个CDF严格大于它的候选。"""
    ids, probabilities = distribution(logits, temperature, top_k)
    cdf = probabilities.cumsum(dim=-1).clamp_max_(1)
    cdf[-1] = 1  # FP32累积舍入不能让末尾抽样落出候选集。
    index = torch.searchsorted(cdf, uniform, right=True).clamp_max(ids.numel() - 1)
    return ids.gather(0, index.reshape(1)).squeeze(0)


class RequestSampler:
    """位置0是Prefill首Token；uniform只读索引，前瞻/丢弃不推进共享游标。"""

    def __init__(self, max_tokens, vocab_size, device, *, temperature=0.0, top_k=-1, seed=None):
        limits = torch.finfo(torch.float32)
        if not math.isfinite(temperature) or not (temperature == 0 or limits.tiny <= temperature <= limits.max):
            raise ValueError("temperature须为0（greedy）或FP32正规正数范围的有限值")
        if type(top_k) is not int or not (top_k == -1 or 1 <= top_k <= vocab_size):
            raise ValueError("top_k须为-1（全词表）或1..vocab_size的整数")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed须为[0, 2^32-1]的整数，或None")
        self.temperature, self.top_k = temperature, top_k
        self.seed = seed if seed is not None or temperature == 0 else secrets.randbits(32)
        self.uniforms = None
        if temperature > 0:
            # CPU私有Generator与固定缓冲长度解耦批次；不触碰CPU/CUDA的全局RNG。
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.uniforms = torch.rand(max_tokens, generator=generator, dtype=torch.float32).to(device)

    def sample(self, logits, position):
        if self.temperature == 0:
            return logits.argmax()
        return sample_uniform(logits, self.temperature, self.top_k, self.uniforms[position])

    def release(self):
        self.uniforms = None


def sample_batch(logits, requests, positions=None, out=None):
    if all(r.sampler.temperature == 0 for r in requests):
        return torch.argmax(logits, dim=-1, out=out)
    positions = positions if positions is not None else [len(r.output_ids) for r in requests]
    tokens = torch.stack([r.sampler.sample(row, position)
                          for row, r, position in zip(logits, requests, positions)])
    if out is not None:
        out.copy_(tokens)
        return out
    return tokens
