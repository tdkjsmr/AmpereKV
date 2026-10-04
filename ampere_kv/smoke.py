"""CUDA 扩展构建与调用链路的冒烟检查。"""

import torch

from ampere_kv import _C


def main() -> None:
    """调用自定义加法 Kernel，并与 PyTorch 结果逐元素比较。"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    left = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    right = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")
    actual = _C.smoke_add(left, right)

    # 同步后再比较，确保异步 Kernel 的运行错误能在这里暴露。
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, left + right, rtol=0.0, atol=0.0)
    print("[PASS] CUDA Extension Smoke")


if __name__ == "__main__":
    main()
