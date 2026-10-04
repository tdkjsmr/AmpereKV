"""单层 BF16/INT8 分页存储；CUDA 读取与当前数值对照位于 paged_decode。"""

import torch

from ampere_kv.block_pool import BlockPool, BlockTable
from ampere_kv.quantization import quantize_kv, dequantize_kv
# 分页 get() 会复制历史；这是参考路径，不是直接读取物理块的 CUDA 内核。


class PagedKVStorage:
    """同一模型层的共享物理 K/V 与块池；所有请求使用相同布局和设备。"""

    def __init__(self, num_kv_heads: int, head_dim: int, num_blocks: int,
                 block_size: int, device="cpu", *, kv_dtype=torch.bfloat16):
        if kv_dtype == torch.int8 and head_dim != 128:
            raise ValueError("当前 INT8 存储固定每头128维，K分成四组32维")
        self._pool = BlockPool(num_blocks)
        # K、V 各自布局为 [物理块数, KV 头数, 每块 Token 数, 每头维度]。
        # 一次分配全部物理空间；申请编号不会再次分配张量，空闲区域内容无效。
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self._key = torch.empty(shape, dtype=kv_dtype, device=device)
        self._value = torch.empty(shape, dtype=kv_dtype, device=device)
        self._key_scale = self._value_scale = None
        if kv_dtype == torch.int8:
            # K 每连续32维一个 scale；V 仍是整个128维一个 scale。
            self._key_scale = torch.empty(shape[:-1] + (4,), dtype=torch.float16, device=device)
            self._value_scale = torch.empty(shape[:-1] + (1,), dtype=torch.float16, device=device)


class PagedKVCache:
    """一个请求的块表与共享存储引用；已分配的块仍由该请求独占。

    单线程使用，不支持共享同一已分配块、引用计数或前缀复用。
    请求结束必须显式 release；丢弃请求对象不会自动向共享池归还编号。
    外部不能修改内部状态，输入 K/V 不能与物理存储共享内存。
    """

    def __init__(self, storage: PagedKVStorage):
        # 这里只复制 Python 引用，不分配或复制 K/V 张量，也不重新创建块池。
        self._pool = storage._pool
        self._key, self._value = storage._key, storage._value
        self._key_scale, self._value_scale = storage._key_scale, storage._value_scale
        self._table = BlockTable(self._pool, self._key.shape[2])

    @property
    def capacity(self) -> int:
        """返回整个共享池的 Token 容量，不保证本请求可全部占用；追加另查空闲块。"""
        return self._key.shape[0] * self._key.shape[2]

    @property
    def length(self) -> int:
        """返回有效 Token 数；仅在 append 正常完成后才保证对应数据完整。"""
        return self._table.length

    def reserve(self, num_tokens: int) -> tuple[int, ...]:
        """只登记尾部新增 Token 所需的块并返回最新块表，不写入任何 K/V。

        批量路径用它把"更新块表"与"融合写入"拆开：B 个请求各调用一次，再由一个批量内核一次写完。
        有效长度会随之增长，所以调用方必须先记下旧长度作为写入起点；登记后写入失败不自动回滚。
        """
        self._table.append_tokens(num_tokens)
        return self._table.block_ids

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor, *, fused: bool = False) -> torch.Tensor | None:
        """追加BF16 K/V；融合分支返回GPU块表供Decode复用，参考分支返回None。"""
        start = self.length
        if fused:
            # 模型内部快路径：GPU 异步失败后本次请求必须丢弃，不能继续用已登记的块表。
            # 支持范围由 CUDA 写入入口校验，这里不重复。
            from ampere_kv import _C
            quantized = self._key_scale is not None
            write = _C.quantize_write if quantized else _C.bf16_write
            key, value = key.contiguous(), value.contiguous()
            self._table.append_tokens(key.shape[2])
            table = torch.tensor(self._table.block_ids, dtype=torch.long, device=key.device)
            if quantized:
                write(key, value, self._key, self._value, self._key_scale, self._value_scale, table, start)
            else:
                write(key, value, self._key, self._value, table, start)
            return table
        if self._key_scale is not None:
            # 参考量化路径：K 临时合并 Token 与组维，只沿 32 维量化，不混合不同 Token 的分量。
            shape = key.shape
            key, key_scale = quantize_kv(key.reshape(1, shape[1], shape[2] * 4, 32))
            key = key.reshape(shape)
            key_scale = key_scale.reshape(1, shape[1], shape[2], 4)
            value, value_scale = quantize_kv(value)
        # 参考路径逐 Token 复制，只求语义清晰；不提供失败回滚，调用方应丢弃本次缓存状态。
        self._table.append_tokens(key.shape[2])
        for index in range(key.shape[2]):
            physical_block, offset = self._table.locate(start + index)
            self._key[physical_block, :, offset, :].copy_(key[0, :, index, :])
            self._value[physical_block, :, offset, :].copy_(value[0, :, index, :])
            if self._key_scale is not None:
                self._key_scale[physical_block, :, offset, :].copy_(key_scale[0, :, index, :])
                self._value_scale[physical_block, :, offset, :].copy_(value_scale[0, :, index, :])

    @torch.no_grad()
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑顺序复制有效 K/V，返回 [1, KV 头数, 有效 Token 数, 每头维度]。

        INT8 还原为 FP32 供参考 Attention 使用；返回独立副本，高性能路径不应这样读回历史。
        """
        if self.length == 0:
            shape = (1, self._key.shape[1], 0, self._key.shape[3])
            dtype = torch.float32 if self._key_scale is not None else torch.bfloat16
            return self._key.new_empty(shape, dtype=dtype), self._value.new_empty(shape, dtype=dtype)
        if self._key_scale is None:
            # BF16 批量读回：一次算出展平行号，K 与 V 共用，避免把块表上传两遍。
            # 行号 = (物理块 · KV头 + 头) · 块大小 + 块内位；索引只由有效长度生成，所以尾块
            # 未写部分取不到，物理块连不连续也无关。逐 Token 切片再 stack 会为一段历史造出
            # 长度份视图和一次 L 路拼接拷贝，分块 Prefill 下每层每个张量都要付一遍。
            blocks, heads, size = self._key.shape[:3]
            positions = torch.arange(self.length, device=self._key.device)
            physical = torch.tensor(self._table.block_ids, dtype=torch.long,
                                    device=self._key.device)[positions // size]
            head_range = torch.arange(heads, device=self._key.device)[:, None]
            rows = ((physical[None, :] * heads + head_range) * size + (positions % size)[None, :]).reshape(-1)
            return tuple(tensor.reshape(-1, tensor.shape[-1]).index_select(0, rows)
                         .view(1, heads, self.length, -1) for tensor in (self._key, self._value))
        # INT8 参考路径按逻辑位置收集整数与 scale，再以 FP32 反量化。
        slots = [self._table.locate(position) for position in range(self.length)]

        def gather(tensor):
            return torch.stack([tensor[block, :, offset, :] for block, offset in slots], dim=1).unsqueeze(0)

        key, value = gather(self._key), gather(self._value)
        # 按相同顺序拆组反量化，再还原每个 Token 的 128 维；不展开 GQA 头。
        grouped_key = key.reshape(1, key.shape[1], self.length * 4, 32)
        scales = gather(self._key_scale).reshape(1, key.shape[1], self.length * 4, 1)
        return (dequantize_kv(grouped_key, scales).reshape(key.shape),
                dequantize_kv(value, gather(self._value_scale)))

    def release(self) -> None:
        """归还请求占用的块；保留物理张量，不清零旧数据，不释放底层显存。"""
        self._table.release()


def main() -> None:
    """检查分组 K 的物理布局与读回；CPU 执行，不调用 CUDA 或模型。"""
    storage = PagedKVStorage(2, 128, 2, 16, kv_dtype=torch.int8)
    cache = PagedKVCache(storage)
    tensors = (cache._key, cache._value, cache._key_scale, cache._value_scale)
    addresses = tuple(t.data_ptr() for t in tensors)
    # 先归还0再归还1，令逻辑块顺序为(1, 0)，检查跨块寻址而非连续巧合。
    held = [storage._pool.allocate() for _ in range(2)]
    for block in held:
        storage._pool.free(block)
    generator = torch.Generator().manual_seed(3)
    key = torch.randn(1, 2, 17, 128, generator=generator)
    key *= torch.tensor([1., 10., 0.1, 100.]).repeat_interleave(32)
    key = key.to(torch.bfloat16)
    value = torch.randn(1, 2, 17, 128, generator=generator).to(torch.bfloat16)
    # 独立按切片逐组量化，避免参考也用同一 reshape 顺序而掩盖布局错误。
    groups = [quantize_kv(key[..., start:start + 32]) for start in range(0, 128, 32)]
    kd = torch.cat([data for data, scale in groups], dim=-1)
    ks = torch.cat([scale for data, scale in groups], dim=-1)
    vd, vs = quantize_kv(value)
    expected_key = torch.cat([data.float() * scale.float() for data, scale in groups], dim=-1)
    try:
        for start, end in ((0, 15), (15, 17)):
            cache.append(key[:, :, start:end], value[:, :, start:end])
            for tensor, reference in zip(tensors, (kd, vd, ks, vs)):
                for token in range(end):
                    block, offset = cache._table.locate(token)
                    assert torch.equal(tensor[block, :, offset], reference[0, :, token])
            actual_key, actual_value = cache.get()
            torch.testing.assert_close(actual_key, expected_key[:, :, :end], rtol=0, atol=0)
            torch.testing.assert_close(actual_value, (vd.float() * vs.float())[:, :, :end], rtol=0, atol=0)
            assert cache.length == end and tuple(t.data_ptr() for t in tensors) == addresses
            print(f"[PASS] 分组K存储：长度={end}，整数与scale布局、FP32读回及地址一致")
        assert cache._table.block_ids == (1, 0)
    finally:
        cache.release()
    assert storage._pool.num_free_blocks == 2
    print("[PASS] CPU分组K存储与跨块追加通过；K scale末维4，V末维1；未验证新布局CUDA、模型或性能")


if __name__ == "__main__":
    main()
