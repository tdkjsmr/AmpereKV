#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <climits>
#include <vector>

namespace ampere_sampling {
using Key = unsigned long long;
constexpr int kThreads = 256, kItems = 8, kSegment = kThreads * kItems;

__device__ Key make_key(float score, unsigned id) {
  unsigned bits = __float_as_uint(score == 0.0f ? 0.0f : score);
  unsigned ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
  return (static_cast<Key>(ordered) << 32) | static_cast<unsigned>(~id);
}

__device__ Key warp_max(Key value) {
  for (int offset = 16; offset; offset /= 2) {
    Key other = __shfl_down_sync(0xffffffff, value, offset);
    value = value > other ? value : other;
  }
  return value;
}

__device__ Key block_max(Key value, Key* scratch) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_max(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = warp_max(lane < 8 ? scratch[lane] : 0);
    if (lane == 0) scratch[8] = value;
  }
  __syncthreads();
  // 下一轮先写scratch[0..7]，到第一次屏障之后才覆盖[8]，所有线程能读到旧赢家。
  return scratch[8];
}

template <typename T>
__global__ void segment_topk(const T* logits, Key* partial, int vocab, int stride, int segments, int k) {
  __shared__ Key scratch[9];
  Key keys[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    int column = blockIdx.x * kSegment + threadIdx.x + i * kThreads;
    keys[i] = column < vocab ? make_key(static_cast<float>(logits[static_cast<int64_t>(blockIdx.y) * stride + column]), column) : 0;
  }
#pragma unroll
  for (int i = 1; i < kItems; ++i) {
#pragma unroll
    for (int j = i; j > 0; --j) {
      if (keys[j] > keys[j - 1]) {
        Key temp = keys[j]; keys[j] = keys[j - 1]; keys[j - 1] = temp;
      }
    }
  }
  for (int rank = 0; rank < k; ++rank) {
    Key best = block_max(keys[0], scratch);
    if (threadIdx.x == 0) partial[(blockIdx.y * segments + blockIdx.x) * k + rank] = best;
    if (keys[0] == best) {
#pragma unroll
      for (int i = 1; i < kItems; ++i) keys[i - 1] = keys[i];
      keys[kItems - 1] = 0;
    }
  }
}

__global__ void merge_topk(const Key* partial, float* values, int64_t* ids, int segments, int k) {
  __shared__ Key scratch[9];
  int segment = threadIdx.x, head = 0;
  const Key* row = partial + blockIdx.x * segments * k;
  Key current = segment < segments ? row[segment * k] : 0;
  // 各段已排序；每次只比较队首，保留K项即可覆盖全局Top-K。
  for (int rank = 0; rank < k; ++rank) {
    Key best = block_max(current, scratch);
    if (threadIdx.x == 0) {
      unsigned ordered = static_cast<unsigned>(best >> 32);
      values[blockIdx.x * k + rank] = __uint_as_float(ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered);
      ids[blockIdx.x * k + rank] = static_cast<unsigned>(~best);
    }
    if (current == best) {
      ++head;
      current = segment < segments && head < k ? row[segment * k + head] : 0;
    }
  }
}

__global__ void sample_candidates(const float* values, const int64_t* ids, const float* temperatures,
                                  const float* uniforms, int64_t* output, float* probs, float* cdf, int k) {
  __shared__ float weights[128];
  int lane = threadIdx.x, row = blockIdx.x;
  float local[4], sum = 0, inverse_temperature = __fdiv_rn(1.0f, temperatures[row]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int rank = lane + 32 * i;
    local[i] = rank < k ? expf((values[row * k + rank] - values[row * k]) * inverse_temperature) : 0;
    sum += local[i];
  }
  for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
  float inverse_sum = __fdiv_rn(1.0f, __shfl_sync(0xffffffff, sum, 0));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int rank = lane + 32 * i;
    if (rank < k) weights[rank] = probs[row * k + rank] = local[i] * inverse_sum;
  }
  __syncwarp();
  if (lane == 0) {
    // 按已输出的FP32概率实际总量归一化，不能只给末项补1。
    double total = 0, prefix = 0;
    for (int i = 0; i < k; ++i) total += weights[i];
    int chosen = k - 1;
    bool found = false;
    for (int i = 0; i < k; ++i) {
      prefix += weights[i];
      float cumulative = static_cast<float>(prefix / total);
      cdf[row * k + i] = cumulative;
      if (!found && uniforms[row] < cumulative) { chosen = i; found = true; }
    }
    output[row] = ids[row * k + chosen];
  }
}
}  // 采样内核命名空间

std::vector<at::Tensor> sampling_topk_cuda(const at::Tensor& logits, int64_t k) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.stride(1) == 1, "logits须为末维连续的CUDA[B,V]");
  TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16, "logits须为FP32/BF16");
  TORCH_CHECK((logits.size(0) == 1 || logits.size(0) == 4) && (k == 40 || k == 128), "支持B1/B4、K40/128");
  TORCH_CHECK(logits.size(1) >= k && logits.size(1) <= 524288 && logits.stride(0) <= INT_MAX, "V须在[K,524288]且行跨度不超int32");
  const c10::cuda::CUDAGuard guard(logits.device());
  int batch = logits.size(0), vocab = logits.size(1), segments = (vocab + 2047) / 2048;
  auto partial = at::empty({batch, segments, k}, logits.options().dtype(at::kLong));
  auto values = at::empty({batch, k}, logits.options().dtype(at::kFloat));
  auto ids = at::empty({batch, k}, logits.options().dtype(at::kLong));
  auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
  auto keys = reinterpret_cast<ampere_sampling::Key*>(partial.data_ptr<int64_t>());
  dim3 grid(segments, batch);
  if (logits.scalar_type() == at::kFloat)
    ampere_sampling::segment_topk<<<grid, 256, 0, stream.stream()>>>(logits.data_ptr<float>(), keys, vocab, logits.stride(0), segments, k);
  else
    ampere_sampling::segment_topk<<<grid, 256, 0, stream.stream()>>>(logits.data_ptr<c10::BFloat16>(), keys, vocab, logits.stride(0), segments, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ampere_sampling::merge_topk<<<batch, 256, 0, stream.stream()>>>(keys, values.data_ptr<float>(), ids.data_ptr<int64_t>(), segments, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, ids};
}

std::vector<at::Tensor> sampling_candidates_cuda(const at::Tensor& values, const at::Tensor& ids,
    const at::Tensor& temperatures, const at::Tensor& uniforms, const at::Tensor& output) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 2 && values.is_contiguous() && values.scalar_type() == at::kFloat,
              "候选分数须为连续CUDA FP32[B,K]");
  int batch = values.size(0), k = values.size(1);
  TORCH_CHECK((batch == 1 || batch == 4) && (k == 40 || k == 128), "支持B1/B4、K40/128");
  TORCH_CHECK(ids.device() == values.device() && ids.sizes() == values.sizes() && ids.is_contiguous()
              && ids.scalar_type() == at::kLong, "候选ID须为同设备int64[B,K]");
  for (const auto& tensor : {temperatures, uniforms, output})
    TORCH_CHECK(tensor.device() == values.device() && tensor.dim() == 1 && tensor.numel() == batch && tensor.is_contiguous(),
                "温度、uniform和输出须为同设备连续[B]");
  TORCH_CHECK(temperatures.scalar_type() == at::kFloat && uniforms.scalar_type() == at::kFloat
              && output.scalar_type() == at::kLong, "温度/uniform须FP32，输出须int64");
  const c10::cuda::CUDAGuard guard(values.device());
  auto probs = at::empty_like(values), cdf = at::empty_like(values);
  auto stream = c10::cuda::getCurrentCUDAStream(values.get_device());
  ampere_sampling::sample_candidates<<<batch, 32, 0, stream.stream()>>>(values.data_ptr<float>(), ids.data_ptr<int64_t>(),
      temperatures.data_ptr<float>(), uniforms.data_ptr<float>(), output.data_ptr<int64_t>(), probs.data_ptr<float>(), cdf.data_ptr<float>(), k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, cdf};
}
