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


def cumulative_probabilities(probabilities):
    """用FP64累计并按实际总量归一化，零概率尾部保持平坦。"""
    prefix = probabilities.cumsum(dim=-1, dtype=torch.float64)
    cdf = (prefix / prefix[..., -1:]).float()
    cdf[..., -1] = 1
    return cdf


def sample_uniform(logits, temperature, top_k, uniform):
    """给定[0,1)的uniform，取首个CDF严格大于它的候选。"""
    ids, probabilities = distribution(logits, temperature, top_k)
    cdf = cumulative_probabilities(probabilities)
    index = torch.searchsorted(cdf, uniform, right=True).clamp_max(ids.numel() - 1)
    return ids.gather(0, index.reshape(1)).squeeze(0)


def cuda_topk(logits, top_k):
    from ampere_kv import _C
    return _C.sampling_topk(logits, top_k)


def cuda_candidates(values, ids, temperatures, uniforms, out):
    from ampere_kv import _C
    return _C.sampling_candidates(values, ids, temperatures, uniforms, out)


def cuda_sample(logits, temperatures, uniforms, top_k, out=None):
    values, ids = cuda_topk(logits, top_k)
    if out is None:
        out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    cuda_candidates(values, ids, temperatures, uniforms, out)
    return out


class RequestSampler:
    """位置0是Prefill首Token；uniform只读索引，前瞻/丢弃不推进共享游标。"""

    def __init__(self, max_tokens, vocab_size, device, *, temperature=0.0, top_k=-1, seed=None,
                 sampling_backend="reference", structured=None):
        limits = torch.finfo(torch.float32)
        if not math.isfinite(temperature) or not (temperature == 0 or limits.tiny <= temperature <= limits.max):
            raise ValueError("temperature须为0（greedy）或FP32正规正数范围的有限值")
        if type(top_k) is not int or not (top_k == -1 or 1 <= top_k <= vocab_size):
            raise ValueError("top_k须为-1（全词表）或1..vocab_size的整数")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed须为[0, 2^32-1]的整数，或None")
        if sampling_backend not in ("reference", "cuda"):
            raise ValueError("sampling_backend须为reference或cuda")
        self.backend = sampling_backend
        self.structured = structured
        self.temperature, self.top_k = temperature, top_k
        self.seed = seed if seed is not None or temperature == 0 else secrets.randbits(32)
        self.uniforms = None
        self.temperature_device = None
        if temperature > 0:
            # CPU私有Generator与固定缓冲长度解耦批次；不触碰CPU/CUDA的全局RNG。
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.uniforms = torch.rand(max_tokens, generator=generator, dtype=torch.float32).to(device)
            if sampling_backend == "cuda" and top_k in (40, 128):
                self.temperature_device = torch.tensor([temperature], dtype=torch.float32, device=device)

    def sample(self, logits, position, *, sampling_backend=None):
        if self.structured is not None:
            self.structured.mask(logits)
        if self.temperature == 0:
            return logits.argmax()
        if (sampling_backend or self.backend) == "cuda" and self.top_k in (40, 128) and logits.numel() <= 524288:
            return cuda_sample(logits[None], self.temperature_device, self.uniforms[position:position + 1], self.top_k)[0]
        return sample_uniform(logits, self.temperature, self.top_k, self.uniforms[position])

    def release(self):
        if self.structured is not None:
            self.structured.release()
            self.structured = None
        self.uniforms = None
        self.temperature_device = None


def sample_batch(logits, requests, positions=None, out=None):
    if all(r.sampler.temperature == 0 for r in requests):
        for row, r in zip(logits, requests):
            if r.sampler.structured is not None:
                r.sampler.structured.mask(row)
        return torch.argmax(logits, dim=-1, out=out)
    positions = positions if positions is not None else [len(r.output_ids) for r in requests]
    states = [r.sampler for r in requests]
    k = states[0].top_k
    if len(states) in (1, 4) and k in (40, 128) and logits.shape[1] <= 524288 and all(
            s.backend == "cuda" and s.top_k == k and s.temperature > 0 for s in states):
        temperatures = torch.cat([s.temperature_device for s in states])
        uniforms = torch.stack([s.uniforms[p] for s, p in zip(states, positions)])
        return cuda_sample(logits, temperatures, uniforms, k, out)
    tokens = torch.stack([r.sampler.sample(row, position, sampling_backend="reference")
                          for row, r, position in zip(logits, requests, positions)])
    if out is not None:
        out.copy_(tokens)
        return out
    return tokens
