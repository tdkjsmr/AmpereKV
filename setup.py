"""构建 AmpereKV 最小 CUDA Extension。"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="ampere-kv-engine",
    version="2.0.1",
    packages=["ampere_kv"],
    extras_require={"structured": ["xgrammar==0.1.18"]},
    ext_modules=[
        CUDAExtension(
            name="ampere_kv._C",
            # setuptools 要求源码使用相对于 setup.py 的正斜杠路径。
            sources=["csrc/bindings.cpp", "csrc/smoke_cuda.cu", "csrc/paged_decode.cu", "csrc/sampling.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_86,code=sm_86",
                    # 保留源码行号用于性能分析，不关闭优化。
                    "--generate-line-info",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
