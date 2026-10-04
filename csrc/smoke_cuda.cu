#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
__global__ void smoke_add_kernel(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ output,
    int64_t num_elements) {
  // 每个 CUDA Thread 只负责一个元素。
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_elements) {
    output[index] = left[index] + right[index];
  }
}

void check_smoke_inputs(
    const at::Tensor& left,
    const at::Tensor& right) {
  TORCH_CHECK(left.is_cuda(), "smoke_add: left 必须位于 CUDA Device");
  TORCH_CHECK(right.is_cuda(), "smoke_add: right 必须位于 CUDA Device");
  TORCH_CHECK(
      left.device() == right.device(),
      "smoke_add: 两个输入必须位于同一 CUDA Device");
  TORCH_CHECK(
      left.sizes() == right.sizes(),
      "smoke_add: 两个输入的形状必须一致");
  TORCH_CHECK(
      left.scalar_type() == right.scalar_type(),
      "smoke_add: 两个输入的 dtype 必须一致");
  TORCH_CHECK(
      left.is_contiguous() && right.is_contiguous(),
      "smoke_add: 只接受连续 Tensor");
}

}  // namespace

at::Tensor smoke_add_cuda(
    const at::Tensor& left,
    const at::Tensor& right) {
  check_smoke_inputs(left, right);

  // 保证输出分配和 Kernel 启动都发生在输入所在的 GPU 上。
  const c10::cuda::CUDAGuard device_guard(left.device());
  at::Tensor output = at::empty_like(left);
  const int64_t num_elements = left.numel();

  // 空 Tensor 没有需要计算的元素，直接返回同形状输出。
  if (num_elements == 0) {
    return output;
  }

  const int64_t blocks =
      (num_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(left.get_device()).stream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      left.scalar_type(),
      "ampere_kv_smoke_add_cuda",
      [&] {
        smoke_add_kernel<scalar_t><<<
            static_cast<unsigned int>(blocks),
            kThreadsPerBlock,
            0,
            stream>>>(
            left.data_ptr<scalar_t>(),
            right.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements);
      });

  // 这里只检查启动错误；同步由 Python 调用端完成。
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
