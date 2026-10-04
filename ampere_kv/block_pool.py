"""最小块编号池与单请求块表：只管理元数据，不分配张量或参与 Attention。"""


class BlockPool:
    """单线程、独占块的编号管理器；不支持共享块、引用计数或请求归属检查。

    块编号只在本池内有效。调用方必须保管当前分配结果，释放后丢弃旧编号；
    编号复用后，本池无法区分合法调用与拿旧编号再次释放的错误调用。
    """

    def __init__(self, num_blocks: int):
        # 列表尾部作为栈顶：初次按 0、1、2……分配，归还的块优先复用。
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        # 下标就是块编号；标记用来拒绝未分配块的释放，避免重复加入空闲列表。
        self._allocated = [False] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        """返回剩余空闲块数；不是剩余 Token 数或显存字节数。"""
        return len(self._free_blocks)

    def allocate(self) -> int:
        """取出一个空闲块编号；耗尽时失败，不自动扩容或抢占其他块。"""
        if not self._free_blocks:
            raise RuntimeError("块池已耗尽，没有空闲块")
        block_id = self._free_blocks.pop()
        self._allocated[block_id] = True
        return block_id

    def free(self, block_id: int) -> None:
        """归还一个已分配编号；只回收编号，不清零任何 K/V。"""
        if not self._allocated[block_id]:
            raise ValueError("不能释放未分配或已经释放的块")
        self._free_blocks.append(block_id)
        self._allocated[block_id] = False


class BlockTable:
    """一个请求独占的有序块表；长度只表示登记的 Token 数，不证明 K/V 已写入。

    仅用于单线程。表中块必须由本表统一释放，外部不能直接向池归还这些编号。
    同一块池的各请求应使用相同块大小；当前池只管理编号，尚无实际存储布局。
    """

    def __init__(self, pool: BlockPool, block_size: int):
        self._pool = pool
        self._block_size = block_size
        self._block_ids: list[int] = []
        self._length = 0
    @property
    def length(self) -> int:
        """返回已登记的有效 Token 数，不包含最后一块的空闲位置。"""
        return self._length

    @property
    def block_ids(self) -> tuple[int, ...]:
        """按逻辑块顺序返回物理编号快照，避免调用方直接修改内部列表。"""
        return tuple(self._block_ids)

    def append_tokens(self, num_tokens: int) -> None:
        """登记尾部新增 Token；只申请缺少的块，不搬运或写入任何 K/V。"""
        new_length = self._length + num_tokens
        # 向上取整：块大小为 4 时，长度 4 需要 1 块，长度 5 才需要 2 块。
        required = (new_length + self._block_size - 1) // self._block_size
        additional = required - len(self._block_ids)
        # 先检查全部需求，避免只分配一部分后才发现耗尽；不提供内存异常下的事务回滚。
        if additional > self._pool.num_free_blocks:
            raise RuntimeError("空闲块不足，本次 Token 追加未执行")
        for _ in range(additional):
            self._block_ids.append(self._pool.allocate())
        self._length = new_length

    def locate(self, token_position: int) -> tuple[int, int]:
        """将从 0 开始的有效 Token 位置映射为（物理块编号，块内偏移）。

        只查询元数据，不证明对应 K/V 已写入；返回值不是显存地址。
        请求释放或块表重新使用后，调用方不能继续使用旧映射。
        """
        # 按有效长度检查，而不是按已分配容量检查，拒绝末块未使用的尾部。
        if type(token_position) is not int or not 0 <= token_position < self._length:
            raise ValueError("Token 位置必须是有效长度范围内的整数")
        # 商确定请求内的逻辑块，余数确定块内偏移；再查表得到物理块编号。
        logical_block, offset = divmod(token_position, self._block_size)
        return self._block_ids[logical_block], offset

    def release(self) -> None:
        """请求结束后归还全部块并重置长度；空表重复调用无操作，可重新使用。"""
        for block_id in self._block_ids:
            self._pool.free(block_id)
        # 清空持有的旧编号，防止下一次 release 再次释放已经归还的块。
        self._block_ids.clear()
        self._length = 0
