#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/macros/Macros.h>
#include <type_traits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>

namespace {
constexpr int kDim = 128;
constexpr int kBlockSize = 16;
constexpr int kSegmentSize = 64;  // 与此前 256 段长做单变量实验，不代表最终最优值。

// 每线程连续四维的打包视图：BF16 走 8 字节，INT8 走 4 字节；起始地址都对齐到位宽。
// 行首是 128 的整数倍，4*lane 再按元素宽度对齐，因此 reinterpret 后的宽载入合法。
union Bf16x4 { float2 wide; __nv_bfloat162 pair[2]; };
union I8x4 { int32_t wide; int8_t value[4]; };

__device__ __forceinline__ void load4(const c10::BFloat16* src, float (&out)[4]) {
  Bf16x4 packed;
  packed.wide = *reinterpret_cast<const float2*>(src);
  const float2 low = __bfloat1622float2(packed.pair[0]);
  const float2 high = __bfloat1622float2(packed.pair[1]);
  out[0] = low.x; out[1] = low.y; out[2] = high.x; out[3] = high.y;
}

__device__ __forceinline__ void load4(const int8_t* src, float (&out)[4]) {
  I8x4 packed;
  packed.wide = *reinterpret_cast<const int32_t*>(src);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = static_cast<float>(packed.value[i]);
}

// 每个(Token, KV头)一个warp；Prefill批量写入，Decode复用同一内核。
// 调用方保证输入有限且scale可用FP16表示；这是模型内部快路径，不清洗外部数据。
__device__ float write_scale(float maximum) {
  const float raw = maximum == 0.0f ? 1.0f : fmaxf(__fdiv_rn(maximum, 127.0f), 0x1p-14f);
  // 必须先舍入到实际保存的FP16 scale，再用同一scale量化。
  return static_cast<float>(static_cast<c10::Half>(raw));
}

__global__ void quantize_write_kernel(
    const c10::BFloat16* key, const c10::BFloat16* value,
    int8_t* output_key, int8_t* output_value, c10::Half* key_scale,
    c10::Half* value_scale, const int64_t* table,
    int heads, int64_t start, int tokens, int64_t blocks) {
  const int lane = threadIdx.x;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int64_t position = start + token;
  const int64_t block = table[position / kBlockSize];
  // 不回传GPU标量；非法页号在设备端终止，不能越界写入其他缓存。
  CUDA_KERNEL_ASSERT(block >= 0 && block < blocks);
  const int offset = position % kBlockSize;
  const int64_t slot = (static_cast<int64_t>(block) * heads + head) * kBlockSize + offset;
  const int64_t input = (static_cast<int64_t>(head) * tokens + token) * kDim;
  float v[4];
  float vmax = 0.0f;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int dim = group * 32 + lane;
    const float k = static_cast<float>(key[input + dim]);
    v[group] = static_cast<float>(value[input + dim]);
    vmax = fmaxf(vmax, fabsf(v[group]));
    float kmax = fabsf(k);
    for (int shift = 16; shift > 0; shift /= 2)
      kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffff, kmax, shift));
    const float scale = write_scale(kmax);
    if (lane == 0) key_scale[slot * 4 + group] = static_cast<c10::Half>(scale);
    const float rounded = nearbyintf(__fdiv_rn(k, scale));  // 最近偶数舍入，与torch.round一致。
    output_key[slot * kDim + dim] = static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, rounded)));
  }
  for (int shift = 16; shift > 0; shift /= 2)
    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, shift));
  const float scale = write_scale(vmax);
  if (lane == 0) value_scale[slot] = static_cast<c10::Half>(scale);
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const float rounded = nearbyintf(__fdiv_rn(v[group], scale));
    output_value[slot * kDim + group * 32 + lane] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, rounded)));
  }
}

// BF16 分页写入：布局与分工同量化写入，每(Token, KV头)一个warp，32线程覆盖128维。
// 只搬运 BF16，不量化、不改变数值；Prefill 整段与 Decode 单 Token 共用同一内核。
__global__ void bf16_write_kernel(
    const c10::BFloat16* key, const c10::BFloat16* value,
    c10::BFloat16* output_key, c10::BFloat16* output_value,
    const int64_t* table, int heads, int64_t start, int tokens, int64_t blocks) {
  const int lane = threadIdx.x;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int64_t position = start + token;
  const int64_t block = table[position / kBlockSize];
  // 不回传GPU标量；非法页号在设备端终止，不能越界写入其他缓存。
  CUDA_KERNEL_ASSERT(block >= 0 && block < blocks);
  const int offset = position % kBlockSize;
  const int64_t slot = (static_cast<int64_t>(block) * heads + head) * kBlockSize + offset;
  const int64_t input = (static_cast<int64_t>(head) * tokens + token) * kDim;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int dim = group * 32 + lane;
    // 按位复制，不经 FP32 往返；写入值必须与输入逐位相同。
    output_key[slot * kDim + dim] = key[input + dim];
    output_value[slot * kDim + dim] = value[input + dim];
  }
}

// 一个线程块的四个 warp 分别处理段内不同 Token；每线程负责连续四维，用打包载入一次取回。
// 不创建连续历史副本，不展开 GQA 的 K/V，不使用 Tensor Core 或异步搬运。
// 只实例化 BF16 与 INT8 两种读取方式，归约和分段合并共用；不是通用类型派发框架。
// V1 的收益在依赖链而不在带宽：本长度下每 SM 只有 1~25 个 warp，掩盖不了流水线气泡，
// 所以缩短 load→scale→乘→累加 的链长比少读字节更关键；INT8 的 K scale 广播已由四次减为一次。
// 本内核保持 V1 原样作为 A/B 基线，四头共享载入见下方的 paged_decode_v3_kernel。
template <typename KV>
__global__ void paged_decode_kernel(
    const c10::BFloat16* query, const KV* key,
    const KV* value, const int64_t* table,
    const c10::Half* key_scale, const c10::Half* value_scale,
    c10::BFloat16* output, float* partials, int64_t length,
    int kv_heads, int group_size, int segments, int64_t blocks) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;  // 固定启动四个完整 warp，shuffle 只在各自 warp 内执行。
  // 只暂存局部结果，不搬运 K/V：四份 FP32 分子、最大值和分母，共 2080 字节。
  __shared__ float local[4][kDim + 2];
  const int query_head = blockIdx.x;
  const int kv_head = query_head / group_size;
  // V1：每线程持有连续四维 4*lane..4*lane+3，一次 8 字节打包载入取代四条标量载入。
  // 固定四元素数组配合展开供编译器标量化；实际寄存器占用以编译结果为准。
  float q[4];
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  load4(query + query_head * kDim + 4 * lane, q);
  // 仅 lane 0 更新段内最大值与分母，其他 lane 通过 shuffle 获取所需数据。
  float maximum = -INFINITY, denominator = 0.0f;

  const int64_t begin = static_cast<int64_t>(blockIdx.y) * kSegmentSize;
  const int64_t end = length < begin + kSegmentSize ? length : begin + kSegmentSize;
  // 交错分工：warp 0 处理 0、4、8……，warp 1 处理 1、5、9……。
  // 完整 64 Token 段中，每个 warp 只串行处理 16 个 Token。
  for (int64_t token = begin + warp; token < end; token += 4) {
    // 布局 [物理块, KV 头, 块内 Token, 维度]；索引使用 64 位避免乘法溢出。
    const int64_t physical = table[token / kBlockSize];
    // 访存前拦截非法块号；这是内部入口唯一的越界保护。块号合法不等于属于当前请求，
    // 归属由块池与块表生命周期保证。
    CUDA_KERNEL_ASSERT(physical >= 0 && physical < blocks);
    const int offset = token % kBlockSize;
    const int64_t index = ((physical * kv_heads + kv_head) * kBlockSize + offset) * kDim + 4 * lane;
    float k_scale = 1.0f, v_scale = 1.0f;
    if constexpr (std::is_same_v<KV, int8_t>) {
      // K scale末维4、V末维1；必须使用KV头而非Query头。
      const int64_t scale_index = (physical * kv_heads + kv_head) * kBlockSize + offset;
      // 本线程的连续四维同属一组（组号 = 4*lane/32 = lane/8），所以 K scale 只需一次广播；
      // 仍由 lane 0..3 各取一组，广播源取 lane/8，不改成 0/8/16/24 装载。
      if (lane < 4) k_scale = static_cast<float>(key_scale[scale_index * 4 + lane]);
      k_scale = __shfl_sync(0xffffffffu, k_scale, lane / 8);
      if (lane == 0) {
        v_scale = static_cast<float>(value_scale[scale_index]);
      }
      v_scale = __shfl_sync(0xffffffffu, v_scale, 0);
    }
    // 相邻 lane 读相邻 8/4 字节，一条指令覆盖整行；scale 已在循环外广播好，链上少三次 shuffle。
    float k4[4];
    load4(key + index, k4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float k = k4[i];
      if constexpr (std::is_same_v<KV, int8_t>) k *= k_scale;  // 先将 K 反量化，再计算点积。
      sum += q[i] * k;
    }
    // 先合并线程内四项，再做 warp 归约；只使用 lane 0 的最终和。
    for (int delta = 16; delta > 0; delta /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    float old_scale = 0.0f, new_weight = 0.0f;
    if (lane == 0) {
      // 浮点加法次序与原树形归约不同，必须重新跑数值与模型对照。
      const float score = sum * rsqrtf(static_cast<float>(kDim));
      // 在线 softmax：最大分数变化时，旧分子和分母都乘同一个缩放系数。
      // 无需保存所有分数；减去最大值避免直接 exp(score) 溢出。
      const float next_maximum = fmaxf(maximum, score);
      old_scale = expf(maximum - next_maximum);
      new_weight = expf(score - next_maximum);
      denominator = denominator * old_scale + new_weight;
      maximum = next_maximum;
    }
    // 分支外由全部 lane 广播系数，不依赖隐式锁步或共享内存。
    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    new_weight = __shfl_sync(0xffffffffu, new_weight, 0);
    float v4[4];
    load4(value + index, v4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = v4[i];
      if constexpr (std::is_same_v<KV, int8_t>) v *= v_scale;
      accumulator[i] = accumulator[i] * old_scale + new_weight * v;
    }
    // Token 循环内仍只有私有状态与 warp shuffle，没有块级屏障。
  }
  // 即使没有分到 Token，也写出初始值：分子和分母为零，最大值为负无穷。
  // 以下四处统一按 4*lane+i 还原成自然维度序，merge_decode_kernel 的接口与编号不变。
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    local[warp][4 * lane + i] = accumulator[i];
  }
  if (lane == 0) {
    local[warp][kDim] = maximum;
    local[warp][kDim + 1] = denominator;
  }
  // 所有线程都必须到达这里；保证四份结果可见后，只有 warp 0 负责合并。
  __syncthreads();
  if (warp != 0) return;

  maximum = -INFINITY;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    maximum = fmaxf(maximum, local[w][kDim]);
  }
  denominator = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) accumulator[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    // 空 warp 不参与指数计算；尤其长度为 1 或末段不足四个 Token 时不可遗漏。
    if (local[w][kDim + 1] > 0.0f) {
      const float scale = expf(local[w][kDim] - maximum);
      denominator += scale * local[w][kDim + 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accumulator[i] += scale * local[w][4 * lane + i];
      }
    }
  }
  if (partials == nullptr) {
    // 单段保留原路径：不分配临时张量，也不启动合并内核。
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output[query_head * kDim + 4 * lane + i] = c10::BFloat16(accumulator[i] / denominator);
    }
  } else {
    // 布局 [Query头, 段, 128维分子 + 最大值 + 分母]，全程保存 FP32。
    // 不能先转 BF16，也不能只保存每段归一化后的输出再取平均。
    const int64_t base = (static_cast<int64_t>(query_head) * segments + blockIdx.y) * (kDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      partials[base + 4 * lane + i] = accumulator[i];
    }
    if (lane == 0) {
      partials[base + kDim] = maximum;
      partials[base + kDim + 1] = denominator;
    }
  }
}

// V3：同一 GQA 组的四个 Query 头放进一个线程块，共用一次 K/V 载入；V1 内核原样保留作 A/B 基线。
// 分工：blockIdx.x 改为组号，块内仍是四个 warp 交错处理段内 Token，每线程负责连续四维 × 四个头。
// 复用发生在寄存器里：块号、K/V 行和两份 scale 每 Token 只取一次，反量化只做一次，四个头各自点积、
// 各自 warp 归约、各自维护 maximum/denominator/累加器，softmax 不跨头混合；每头算术次序与 V1 相同。
// 代价是并行度：同段的块数从 Query 头数降到 KV 头数（32→8），短上下文可能反而变慢，必须实测。
template <typename KV>
__global__ void paged_decode_v3_kernel(
    const c10::BFloat16* query, const KV* key,
    const KV* value, const int64_t* table,
    const c10::Half* key_scale, const c10::Half* value_scale,
    c10::BFloat16* output, float* partials, int64_t length,
    int kv_heads, int group_size, int segments, int64_t blocks) {
  constexpr int kHeads = 4;  // 入口已检查 group_size == 4，一个块恰好覆盖一组。
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  // 四个 warp × 四个头 × (128 维分子 + 最大值 + 分母)：静态共享内存由 2080 增至 8320 字节。
  __shared__ float local[4][kHeads][kDim + 2];
  const int first_head = blockIdx.x * kHeads;
  const int kv_head = first_head / group_size;
  // 十六个 query 分量与十六个 FP32 累加器都留在寄存器，这是四头复用一次载入的前提。
  float q[kHeads][4], accumulator[kHeads][4] = {}, maximum[kHeads], denominator[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    load4(query + (first_head + h) * kDim + 4 * lane, q[h]);
    maximum[h] = -INFINITY;
    denominator[h] = 0.0f;
  }
  const int64_t begin = static_cast<int64_t>(blockIdx.y) * kSegmentSize;
  const int64_t end = length < begin + kSegmentSize ? length : begin + kSegmentSize;
  for (int64_t token = begin + warp; token < end; token += 4) {
    const int64_t physical = table[token / kBlockSize];
    // 与 V1 同一条保护：任何 K/V 或 scale 访存之前拦下非法块号。
    CUDA_KERNEL_ASSERT(physical >= 0 && physical < blocks);
    const int offset = token % kBlockSize;
    const int64_t index = ((physical * kv_heads + kv_head) * kBlockSize + offset) * kDim + 4 * lane;
    float k_scale = 1.0f, v_scale = 1.0f;
    if constexpr (std::is_same_v<KV, int8_t>) {
      const int64_t scale_index = (physical * kv_heads + kv_head) * kBlockSize + offset;
      // 广播链每 Token 只走一遍，服务四个头；V1 是每头各走一遍。
      if (lane < 4) k_scale = static_cast<float>(key_scale[scale_index * 4 + lane]);
      k_scale = __shfl_sync(0xffffffffu, k_scale, lane / 8);
      if (lane == 0) {
        v_scale = static_cast<float>(value_scale[scale_index]);
      }
      v_scale = __shfl_sync(0xffffffffu, v_scale, 0);
    }
    float k4[4];
    load4(key + index, k4);
    if constexpr (std::is_same_v<KV, int8_t>) {
#pragma unroll
      for (int i = 0; i < 4; ++i) k4[i] *= k_scale;  // 反量化一次，四个头共用。
    }
    float sum[kHeads] = {};
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
      for (int i = 0; i < 4; ++i) sum[h] += q[h][i] * k4[i];
      // 每头一条独立的 warp 归约链，台阶次序与 V1 相同；四个头共 20 次 shuffle。
      for (int delta = 16; delta > 0; delta /= 2) sum[h] += __shfl_down_sync(0xffffffffu, sum[h], delta);
    }
    float old_scale[kHeads] = {}, new_weight[kHeads] = {};
    if (lane == 0) {
      // 每头一份在线 softmax；缩放系数只作用于本头的分子与分母。
#pragma unroll
      for (int h = 0; h < kHeads; ++h) {
        const float score = sum[h] * rsqrtf(static_cast<float>(kDim));
        const float next_maximum = fmaxf(maximum[h], score);
        old_scale[h] = expf(maximum[h] - next_maximum);
        new_weight[h] = expf(score - next_maximum);
        denominator[h] = denominator[h] * old_scale[h] + new_weight[h];
        maximum[h] = next_maximum;
      }
    }
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      old_scale[h] = __shfl_sync(0xffffffffu, old_scale[h], 0);
      new_weight[h] = __shfl_sync(0xffffffffu, new_weight[h], 0);
    }
    float v4[4];
    load4(value + index, v4);
    if constexpr (std::is_same_v<KV, int8_t>) {
#pragma unroll
      for (int i = 0; i < 4; ++i) v4[i] *= v_scale;
    }
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
      for (int i = 0; i < 4; ++i) accumulator[h][i] = accumulator[h][i] * old_scale[h] + new_weight[h] * v4[i];
    }
  }
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) local[warp][h][4 * lane + i] = accumulator[h][i];
    if (lane == 0) {
      local[warp][h][kDim] = maximum[h];
      local[warp][h][kDim + 1] = denominator[h];
    }
  }
  __syncthreads();
  if (warp != 0) return;
  // warp 0 逐头合并四个 warp；partial 行号仍按 Query 头，merge_decode_kernel 与接口不变。
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    const int query_head = first_head + h;
    maximum[h] = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; ++w) maximum[h] = fmaxf(maximum[h], local[w][h][kDim]);
    denominator[h] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) accumulator[h][i] = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      // 空 warp 不参与指数计算；长度为 1 或末段不足四个 Token 时不可遗漏。
      if (local[w][h][kDim + 1] > 0.0f) {
        const float scale = expf(local[w][h][kDim] - maximum[h]);
        denominator[h] += scale * local[w][h][kDim + 1];
#pragma unroll
        for (int i = 0; i < 4; ++i) accumulator[h][i] += scale * local[w][h][4 * lane + i];
      }
    }
    if (partials == nullptr) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        output[query_head * kDim + 4 * lane + i] = c10::BFloat16(accumulator[h][i] / denominator[h]);
      }
    } else {
      const int64_t base = (static_cast<int64_t>(query_head) * segments + blockIdx.y) * (kDim + 2);
#pragma unroll
      for (int i = 0; i < 4; ++i) partials[base + 4 * lane + i] = accumulator[h][i];
      if (lane == 0) {
        partials[base + kDim] = maximum[h];
        partials[base + kDim + 1] = denominator[h];
      }
    }
  }
}

// 批量分页 Decode：一次启动同时推进 B 个请求的当前 Token，每个块负责一个 (请求, Query 头)。
// 与 V1 的唯一区别是索引来源：块表是 [B, 每请求最大块数] 的行，各请求的写入起点从设备端 starts
// 读取（每步每请求一个 Token，有效长度即 起点+1）。因此不把任何请求的历史读回连续显存，
// 也不把长度搬回宿主——每步仍只有设备端可见的元数据。
// 每个 (请求, 头) 的算术次序与 V1 逐条相同：同样段长、四 warp 交错、连续四维打包与归约台阶。
// 只实例化 BF16；INT8 多请求按计划推到基础链路稳定之后。分段合并仍用 merge_decode_kernel，
// 只是行号由 Query 头号换成 (请求, 头) 行号。
__global__ void paged_decode_batched_kernel(
    const c10::BFloat16* query, const c10::BFloat16* key, const c10::BFloat16* value,
    const int64_t* table, const int64_t* starts,
    c10::BFloat16* output, float* partials,
    int q_heads, int kv_heads, int group_size, int table_stride, int segments, int64_t blocks) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  __shared__ float local[4][kDim + 2];
  const int row = blockIdx.x;  // 行号 = 请求 * Query 头数 + Query 头，与 [批, 头, 1, 128] 布局一致。
  const int request = row / q_heads;
  const int kv_head = (row % q_heads) / group_size;
  const int64_t start = starts[request];
  // 请求级范围保护，放在任何块表与 K/V 访存之前，且不做宿主同步：起点必须非负，
  // 且本步写入的位置（start 本身）要落在这一行块表能容纳的 Token 数之内。
  // 起点为负时 position 会变成负数，而"槽位小于行宽"这类断言挡不住负下标，会先越界读。
  CUDA_KERNEL_ASSERT(start >= 0 && start < static_cast<int64_t>(table_stride) * kBlockSize);
  const int64_t length = start + 1;  // 每步每请求写入一个 Token，有效长度即起点 + 1。
  const int64_t* row_table = table + static_cast<int64_t>(request) * table_stride;
  float q[4];
  float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  load4(query + static_cast<int64_t>(row) * kDim + 4 * lane, q);
  float maximum = -INFINITY, denominator = 0.0f;

  const int64_t begin = static_cast<int64_t>(blockIdx.y) * kSegmentSize;
  const int64_t end = length < begin + kSegmentSize ? length : begin + kSegmentSize;
  // 超出本请求长度的分段整块空转，只写出全零局部结果，由跨段合并按分母为零跳过。
  for (int64_t token = begin + warp; token < end; token += 4) {
    // token < length <= 行宽 × 16，所以入口那条起点范围保护已经保证这里不会读出行宽。
    const int64_t physical = row_table[token / kBlockSize];
    // 与 V1 同一条保护：任何 K/V 访存之前拦下非法块号。
    CUDA_KERNEL_ASSERT(physical >= 0 && physical < blocks);
    const int64_t index = ((physical * kv_heads + kv_head) * kBlockSize + token % kBlockSize) * kDim + 4 * lane;
    float k4[4];
    load4(key + index, k4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum += q[i] * k4[i];
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffffu, sum, delta);
    float old_scale = 0.0f, new_weight = 0.0f;
    if (lane == 0) {
      const float score = sum * rsqrtf(static_cast<float>(kDim));
      const float next_maximum = fmaxf(maximum, score);
      old_scale = expf(maximum - next_maximum);
      new_weight = expf(score - next_maximum);
      denominator = denominator * old_scale + new_weight;
      maximum = next_maximum;
    }
    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    new_weight = __shfl_sync(0xffffffffu, new_weight, 0);
    float v4[4];
    load4(value + index, v4);
#pragma unroll
    for (int i = 0; i < 4; ++i) accumulator[i] = accumulator[i] * old_scale + new_weight * v4[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) local[warp][4 * lane + i] = accumulator[i];
  if (lane == 0) {
    local[warp][kDim] = maximum;
    local[warp][kDim + 1] = denominator;
  }
  __syncthreads();
  if (warp != 0) return;
  maximum = -INFINITY;
#pragma unroll
  for (int w = 0; w < 4; ++w) maximum = fmaxf(maximum, local[w][kDim]);
  denominator = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) accumulator[i] = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    // 空 warp 与空分段都不参与指数计算；末段不足四个 Token 时同样适用。
    if (local[w][kDim + 1] > 0.0f) {
      const float scale = expf(local[w][kDim] - maximum);
      denominator += scale * local[w][kDim + 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) accumulator[i] += scale * local[w][4 * lane + i];
    }
  }
  if (partials == nullptr) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output[static_cast<int64_t>(row) * kDim + 4 * lane + i] = c10::BFloat16(accumulator[i] / denominator);
    }
  } else {
    const int64_t base = (static_cast<int64_t>(row) * segments + blockIdx.y) * (kDim + 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) partials[base + 4 * lane + i] = accumulator[i];
    if (lane == 0) {
      partials[base + kDim] = maximum;
      partials[base + kDim + 1] = denominator;
    }
  }
}

// 批量 BF16 写入：每个 (请求, KV 头) 一个 warp，把 [B, KV头, Token, 128] 的新 K/V
// 按各自块表写进同一层的共享存储。起点与批量 Decode 内核读同一份设备端 starts，宿主不看任何标量。
__global__ void bf16_write_batched_kernel(
    const c10::BFloat16* key, const c10::BFloat16* value,
    c10::BFloat16* output_key, c10::BFloat16* output_value,
    const int64_t* table, const int64_t* starts,
    int heads, int table_stride, int tokens, int64_t blocks) {
  const int lane = threadIdx.x;  // 固定 32 线程，分工与单请求 bf16_write_kernel 相同。
  const int head = blockIdx.y;
  const int request = blockIdx.x / tokens;
  const int token = blockIdx.x % tokens;
  const int64_t start = starts[request];
  // 与批量 Decode 内核同一条保护，且同样不做宿主同步：起点非负，整段写入范围落在这一行内。
  // 负起点会让 position 变负，而"槽位小于行宽"的断言挡不住负下标，会先越界读。
  CUDA_KERNEL_ASSERT(start >= 0 && start + tokens <= static_cast<int64_t>(table_stride) * kBlockSize);
  const int64_t position = start + token;
  const int64_t block = table[static_cast<int64_t>(request) * table_stride + position / kBlockSize];
  CUDA_KERNEL_ASSERT(block >= 0 && block < blocks);
  const int64_t physical = (block * heads + head) * kBlockSize + position % kBlockSize;
  const int64_t input = ((static_cast<int64_t>(request) * heads + head) * tokens + token) * kDim;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int dim = group * 32 + lane;
    // 按位复制，不经 FP32 往返；与单请求写入内核逐位一致。
    output_key[physical * kDim + dim] = key[input + dim];
    output_value[physical * kDim + dim] = value[input + dim];
  }
}

// 同一 CUDA 流上的第二次启动保证所有局部结果已写好，不需要 CPU 同步。
__global__ void merge_decode_kernel(const float* partials, c10::BFloat16* output, int segments) {
  const int dim = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * segments * (kDim + 2);
  float maximum = -INFINITY;
  for (int segment = 0; segment < segments; ++segment) {
    maximum = fmaxf(maximum, partials[base + static_cast<int64_t>(segment) * (kDim + 2) + kDim]);
  }
  float numerator = 0.0f, denominator = 0.0f;
  for (int segment = 0; segment < segments; ++segment) {
    const int64_t offset = base + static_cast<int64_t>(segment) * (kDim + 2);
    // 各段使用各自最大值计算过指数；重新缩放到共同最大值后才能相加。
    const float scale = expf(partials[offset + kDim] - maximum);
    numerator += scale * partials[offset + dim];
    denominator += scale * partials[offset + kDim + 1];
  }
  output[blockIdx.x * kDim + dim] = c10::BFloat16(numerator / denominator);
}
}  // 匿名命名空间：内部实现不暴露给其他编译单元。

// V1 与 V3 参数表相同，这里只按 v3 选一次内核，避免把启动参数抄四份；不是通用派发框架。
template <typename KV>
static void launch_decode(bool v3, const dim3& grid, cudaStream_t stream, const at::Tensor& query,
                          const at::Tensor& key, const at::Tensor& value, const at::Tensor& table,
                          const c10::Half* key_scale, const c10::Half* value_scale,
                          at::Tensor& output, float* partials, int64_t length, int kv_heads,
                          int group_size, int segments, int64_t blocks) {
  if (v3) {
    paged_decode_v3_kernel<KV><<<grid, 128, 0, stream>>>(query.data_ptr<c10::BFloat16>(), key.data_ptr<KV>(),
        value.data_ptr<KV>(), table.data_ptr<int64_t>(), key_scale, value_scale,
        output.data_ptr<c10::BFloat16>(), partials, length, kv_heads, group_size, segments, blocks);
  } else {
    paged_decode_kernel<KV><<<grid, 128, 0, stream>>>(query.data_ptr<c10::BFloat16>(), key.data_ptr<KV>(),
        value.data_ptr<KV>(), table.data_ptr<int64_t>(), key_scale, value_scale,
        output.data_ptr<c10::BFloat16>(), partials, length, kv_heads, group_size, segments, blocks);
  }
}

// 共用形状检查、工作区与启动；quantized 决定精度，check_table 决定是否做宿主块号检查，
// v3 决定每块服务一个还是同一 GQA 组的四个 Query 头。
static at::Tensor paged_decode_impl(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table,
                            int64_t length, bool quantized, bool check_table,
                            const at::Tensor& key_scale, const at::Tensor& value_scale, bool v3) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && table.is_cuda(), "输入必须全部位于 CUDA");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device() && key.device() == table.device(), "输入必须位于同一设备");
  const auto kv_type = quantized ? at::kChar : at::kBFloat16;
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == kv_type && value.scalar_type() == kv_type, "Q 必须是 BF16，K/V 类型必须匹配所选入口");
  TORCH_CHECK(table.scalar_type() == at::kLong && table.dim() == 1, "块表必须是一维 int64");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && table.is_contiguous(), "分页 Decode 只接受连续布局张量");
  TORCH_CHECK(query.dim() == 4 && query.size(0) == 1 && query.size(2) == 1 && query.size(3) == kDim, "Q 必须是 [1, Query头数, 1, 128]");
  TORCH_CHECK(key.dim() == 4 && key.sizes() == value.sizes() && key.size(0) > 0 && key.size(2) == kBlockSize && key.size(3) == kDim, "K/V 必须是 [物理块数, KV头数, 16, 128]");
  const int64_t q_heads = query.size(1), kv_heads = key.size(1);
  TORCH_CHECK(q_heads > 0 && q_heads <= 1024 && kv_heads > 0 && q_heads % kv_heads == 0, "Query 头数必须不超过 1024 且是 KV 头数的正整数倍");
  // V3 不支持的配置直接报错，不静默换成 V1：这样"这一轮跑完了"本身就是用的四头内核。
  // 需要跑非 4 倍 GQA 的调用方请继续用默认（v3=false）的 V1 入口。
  TORCH_CHECK(!v3 || q_heads == kv_heads * 4, "V3 只支持每组四个 Query 头，其他配置请使用 V1 入口");
  TORCH_CHECK(length > 0 && (length - 1) / kBlockSize < table.numel(), "有效长度必须为正且块表必须足够长");
  if (quantized) {
    TORCH_CHECK(key_scale.is_cuda() && value_scale.is_cuda() && key_scale.device() == query.device() && value_scale.device() == query.device(), "scale 必须与 Q 同一 CUDA 设备");
    TORCH_CHECK(key_scale.scalar_type() == at::kHalf && value_scale.scalar_type() == at::kHalf && key_scale.is_contiguous() && value_scale.is_contiguous(), "scale 必须是连续 FP16 张量");
    TORCH_CHECK(key_scale.dim() == 4 && key_scale.size(0) == key.size(0) && key_scale.size(1) == kv_heads && key_scale.size(2) == kBlockSize && key_scale.size(3) == 4, "K scale必须是 [物理块数, KV头数, 16, 4]，请使用分组K存储");
    TORCH_CHECK(value_scale.dim() == 4 && value_scale.size(0) == key.size(0) && value_scale.size(1) == kv_heads && value_scale.size(2) == kBlockSize && value_scale.size(3) == 1, "V scale必须是 [物理块数, KV头数, 16, 1]");
    // 有效槽位的整数范围与有限正 scale 由已验证的量化写入保证；此处不扫描数值。
    // 未写尾部允许哨兵值，内核只按有效长度读取；此入口不是不可信数据清洗器。
  }
  const int64_t segment_count = (length - 1) / kSegmentSize + 1;
  TORCH_CHECK(segment_count <= 65535, "分段数超过二维 CUDA 网格上限");
  const int segments = static_cast<int>(segment_count);
  const c10::cuda::CUDAGuard guard(query.device());
  if (check_table) {
    // 受检查入口保留宿主块号保护：非法块号同步报错，调用方可以捕获后继续做其他检查。
    const auto used_table = table.narrow(0, 0, (length - 1) / kBlockSize + 1);
    TORCH_CHECK(used_table.min().item<int64_t>() >= 0 && used_table.max().item<int64_t>() < key.size(0), "物理块编号越界");
  }
  // 内部入口跳过宿主标量取回，消除每层每 Token 的 CPU 等待；越界由内核在访存前断言拦截。
  // 设备端断言是异步失败且会毒化上下文，内部路径出现非法块表视为实现错误，本次运行终止。
  auto output = at::empty_like(query);
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  // 长历史才建立局部结果；临时存储分配和额外启动均计入现有调用基线。
  at::Tensor partials;
  float* partial_ptr = nullptr;
  if (segments > 1) {
    partials = at::empty({q_heads, segment_count, kDim + 2}, query.options().dtype(at::kFloat));
    partial_ptr = partials.data_ptr<float>();
  }
  // V3 的横坐标是 GQA 组数，V1 仍是 Query 头数；四个 warp 独立计算、段末合并，跨段合并内核不变。
  const dim3 grid(static_cast<unsigned int>(v3 ? kv_heads : q_heads), static_cast<unsigned int>(segments));
  if (quantized) {
    launch_decode<int8_t>(v3, grid, stream, query, key, value, table,
                          key_scale.data_ptr<c10::Half>(), value_scale.data_ptr<c10::Half>(), output,
                          partial_ptr, length, static_cast<int>(kv_heads),
                          static_cast<int>(q_heads / kv_heads), segments, key.size(0));
  } else {
    launch_decode<c10::BFloat16>(v3, grid, stream, query, key, value, table, nullptr, nullptr, output,
                                 partial_ptr, length, static_cast<int>(kv_heads),
                                 static_cast<int>(q_heads / kv_heads), segments, key.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (segments > 1) {
    merge_decode_kernel<<<static_cast<unsigned int>(q_heads), kDim, 0, stream>>>(
        partial_ptr, output.data_ptr<c10::BFloat16>(), segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

// 只检查CPU可见元数据，不读取GPU标量；块归属由Python块表管理。
void quantize_write_cuda(const at::Tensor& key, const at::Tensor& value,
                         const at::Tensor& output_key, const at::Tensor& output_value,
                         const at::Tensor& key_scale, const at::Tensor& value_scale,
                         const at::Tensor& table, int64_t start) {
  TORCH_CHECK(key.is_cuda() && key.dim() == 4 && key.size(0) == 1 &&
              key.size(2) > 0 && key.size(2) <= 65535 && key.size(3) == kDim &&
              key.size(1) > 0 && key.size(1) <= 1024,
              "写入输入必须为CUDA [1, KV头, Token, 128]");
  for (const auto& tensor : {key, value, output_key, output_value, key_scale, value_scale, table})
    TORCH_CHECK(tensor.device() == key.device() && tensor.is_contiguous(), "写入张量必须同设备且连续");
  TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16 &&
              key.sizes() == value.sizes(), "新K/V必须为同形状BF16");
  TORCH_CHECK(output_key.dim() == 4 && output_key.size(1) == key.size(1) &&
              output_key.size(2) == kBlockSize && output_key.size(3) == kDim &&
              output_key.sizes() == output_value.sizes() &&
              output_key.scalar_type() == at::kChar && output_value.scalar_type() == at::kChar,
              "写入目标必须为INT8 [块数, KV头, 16, 128]");
  TORCH_CHECK(table.scalar_type() == at::kLong && table.dim() == 1 && start >= 0 &&
              start <= table.numel() * kBlockSize - key.size(2), "块表或写入范围无效");
  for (const auto& tensor : {key_scale, value_scale})
    TORCH_CHECK(tensor.scalar_type() == at::kHalf && tensor.dim() == 4 &&
                tensor.size(0) == output_key.size(0) && tensor.size(1) == key.size(1) &&
                tensor.size(2) == kBlockSize, "scale布局或类型错误");
  TORCH_CHECK(key_scale.size(3) == 4 && value_scale.size(3) == 1, "K/V scale末维应为4/1");
  const c10::cuda::CUDAGuard guard(key.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  const dim3 grid(static_cast<unsigned int>(key.size(1)), static_cast<unsigned int>(key.size(2)));
  quantize_write_kernel<<<grid, 32, 0, stream>>>(
      key.data_ptr<c10::BFloat16>(), value.data_ptr<c10::BFloat16>(),
      output_key.data_ptr<int8_t>(), output_value.data_ptr<int8_t>(),
      key_scale.data_ptr<c10::Half>(), value_scale.data_ptr<c10::Half>(),
      table.data_ptr<int64_t>(), static_cast<int>(key.size(1)), start,
      static_cast<int>(key.size(2)), output_key.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 只检查CPU可见元数据，不读取GPU标量；BF16 写入不量化，因此没有 scale 相关约束。
void bf16_write_cuda(const at::Tensor& key, const at::Tensor& value,
                     const at::Tensor& output_key, const at::Tensor& output_value,
                     const at::Tensor& table, int64_t start) {
  TORCH_CHECK(key.is_cuda() && key.dim() == 4 && key.size(0) == 1 &&
              key.size(2) > 0 && key.size(2) <= 65535 && key.size(3) == kDim &&
              key.size(1) > 0 && key.size(1) <= 1024,
              "写入输入必须为CUDA [1, KV头, Token, 128]");
  for (const auto& tensor : {key, value, output_key, output_value, table})
    TORCH_CHECK(tensor.device() == key.device() && tensor.is_contiguous(), "写入张量必须同设备且连续");
  TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16 &&
              key.sizes() == value.sizes(), "新K/V必须为同形状BF16");
  TORCH_CHECK(output_key.dim() == 4 && output_key.size(1) == key.size(1) &&
              output_key.size(2) == kBlockSize && output_key.size(3) == kDim &&
              output_key.sizes() == output_value.sizes() &&
              output_key.scalar_type() == at::kBFloat16 && output_value.scalar_type() == at::kBFloat16,
              "写入目标必须为BF16 [块数, KV头, 16, 128]");
  // 只校验写入范围与块表长度自洽；start 等于旧有效长度、块归属正确由调用方保证。
  TORCH_CHECK(table.scalar_type() == at::kLong && table.dim() == 1 && start >= 0 &&
              start <= table.numel() * kBlockSize - key.size(2), "块表或写入范围无效");
  const c10::cuda::CUDAGuard guard(key.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  const dim3 grid(static_cast<unsigned int>(key.size(1)), static_cast<unsigned int>(key.size(2)));
  bf16_write_kernel<<<grid, 32, 0, stream>>>(
      key.data_ptr<c10::BFloat16>(), value.data_ptr<c10::BFloat16>(),
      output_key.data_ptr<c10::BFloat16>(), output_value.data_ptr<c10::BFloat16>(),
      table.data_ptr<int64_t>(), static_cast<int>(key.size(1)), start,
      static_cast<int>(key.size(2)), output_key.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 受检查入口：供独立算子检查与外部调用；非法块号同步报错，调用方可捕获后继续。
// v3 默认 false，即仍走 V1；V3 只是实验选择，未测量前不替换默认路径。
at::Tensor paged_decode_cuda(const at::Tensor& query, const at::Tensor& key,
                            const at::Tensor& value, const at::Tensor& table, int64_t length, bool v3) {
  return paged_decode_impl(query, key, value, table, length, false, true, at::Tensor(), at::Tensor(), v3);
}

at::Tensor paged_decode_int8_cuda(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value, const at::Tensor& key_scale,
                                 const at::Tensor& value_scale, const at::Tensor& table, int64_t length, bool v3) {
  return paged_decode_impl(query, key, value, table, length, true, true, key_scale, value_scale, v3);
}

// 模型内部入口：不做宿主标量取回，块号改由内核在任何 K/V 或 scale 访存之前保护。
// 形状、类型、设备、长度和块表长度检查仍然无条件执行；非法块表属实现错误，
// 会在后续同步点异步失败并终止本次运行，不提供可恢复的 GPU 错误处理。
at::Tensor paged_decode_internal_cuda(const at::Tensor& query, const at::Tensor& key,
                                     const at::Tensor& value, const at::Tensor& table, int64_t length, bool v3) {
  return paged_decode_impl(query, key, value, table, length, false, false, at::Tensor(), at::Tensor(), v3);
}

at::Tensor paged_decode_int8_internal_cuda(const at::Tensor& query, const at::Tensor& key,
                                          const at::Tensor& value, const at::Tensor& key_scale,
                                          const at::Tensor& value_scale, const at::Tensor& table, int64_t length, bool v3) {
  return paged_decode_impl(query, key, value, table, length, true, false, key_scale, value_scale, v3);
}

// 批量 BF16 写入：table 是 [批, 每请求块数]，starts 是每请求的写入起点（有效长度之前的位置）。
// 起点与块数都在设备端，宿主只检查元数据形状，不取回任何标量。
void bf16_write_batched_cuda(const at::Tensor& key, const at::Tensor& value,
                             const at::Tensor& output_key, const at::Tensor& output_value,
                             const at::Tensor& table, const at::Tensor& starts) {
  TORCH_CHECK(key.is_cuda() && key.dim() == 4 && key.size(0) > 0 && key.size(1) > 0 && key.size(1) <= 1024 &&
              key.size(2) > 0 && key.size(3) == kDim, "批量写入输入必须是 CUDA [批, KV头, Token, 128]");
  TORCH_CHECK(key.sizes() == value.sizes() && key.scalar_type() == at::kBFloat16 &&
              value.scalar_type() == at::kBFloat16, "新 K/V 必须为同形状 BF16");
  TORCH_CHECK(output_key.sizes() == output_value.sizes() && output_key.dim() == 4 &&
              output_key.scalar_type() == at::kBFloat16 && output_key.size(1) == key.size(1) &&
              output_key.size(2) == kBlockSize && output_key.size(3) == kDim,
              "写入目标必须是 BF16 [块数, KV头, 16, 128]");
  TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kLong && table.size(0) == key.size(0) &&
              table.size(1) > 0, "块表必须是二维定宽 int64 [批, 每请求块数]");
  TORCH_CHECK(starts.dim() == 1 && starts.scalar_type() == at::kLong && starts.size(0) == key.size(0),
              "写入起点必须是每请求一个 int64");
  TORCH_CHECK(key.device() == value.device() && key.device() == output_key.device() &&
              key.device() == output_value.device() && key.device() == table.device() &&
              key.device() == starts.device(), "批量写入张量必须位于同一设备");
  for (const auto& tensor : {key, value, output_key, output_value, table, starts})
    TORCH_CHECK(tensor.is_contiguous(), "批量写入只接受连续布局");
  // 每请求的写入范围是否越出其块表行，由内核在读表之前用设备端断言拦截；这里不做宿主扫描，
  // 因为那需要把 starts 搬回 CPU。
  const c10::cuda::CUDAGuard guard(key.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(key.get_device()).stream();
  const dim3 grid(static_cast<unsigned int>(key.size(0) * key.size(2)), static_cast<unsigned int>(key.size(1)));
  bf16_write_batched_kernel<<<grid, 32, 0, stream>>>(
      key.data_ptr<c10::BFloat16>(), value.data_ptr<c10::BFloat16>(),
      output_key.data_ptr<c10::BFloat16>(), output_value.data_ptr<c10::BFloat16>(),
      table.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), static_cast<int>(key.size(1)),
      static_cast<int>(table.size(1)), static_cast<int>(key.size(2)), output_key.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 批量 Decode 入口：一次启动处理 B 个请求各一个 Token，返回与 query 同形状的 [批, Query头, 1, 128]。
// 分段数按块表行宽上界确定，因此不需要读取设备端长度；超出某请求实际长度的分段整块空转，
// 由跨段合并按分母为零跳过，结果与逐请求单独调用 V1 内核的分段方式一致。
at::Tensor paged_decode_batched_cuda(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                                     const at::Tensor& table, const at::Tensor& starts) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && table.is_cuda() && starts.is_cuda(),
              "批量 Decode 输入必须全部位于 CUDA");
  TORCH_CHECK(query.device() == key.device() && key.device() == value.device() &&
              key.device() == table.device() && key.device() == starts.device(), "输入必须位于同一设备");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
              value.scalar_type() == at::kBFloat16, "批量 Decode 只支持 BF16");
  TORCH_CHECK(query.dim() == 4 && query.size(0) > 0 && query.size(1) > 0 && query.size(1) <= 1024 &&
              query.size(2) == 1 && query.size(3) == kDim, "批量 Q 必须是 [批, Query头, 1, 128]");
  TORCH_CHECK(key.sizes() == value.sizes() && key.dim() == 4 && key.size(0) > 0 && key.size(1) > 0 &&
              key.size(2) == kBlockSize && key.size(3) == kDim, "K/V 必须是 [块数, KV头, 16, 128]");
  TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kLong && table.size(0) == query.size(0) &&
              table.size(1) > 0, "块表必须是二维定宽 int64 [批, 每请求块数]");
  TORCH_CHECK(starts.dim() == 1 && starts.scalar_type() == at::kLong && starts.size(0) == query.size(0),
              "写入起点必须是每请求一个 int64");
  const int64_t q_heads = query.size(1), kv_heads = key.size(1);
  TORCH_CHECK(q_heads % kv_heads == 0, "Query 头数必须是 KV 头数的正整数倍");
  for (const auto& tensor : {query, key, value, table, starts})
    TORCH_CHECK(tensor.is_contiguous(), "批量 Decode 只接受连续布局");
  const int64_t rows = query.size(0) * q_heads;
  const int64_t segment_count = (table.size(1) * kBlockSize + kSegmentSize - 1) / kSegmentSize;
  TORCH_CHECK(segment_count <= 65535 && rows <= 65535, "批量网格超过二维 CUDA 网格上限");
  const int segments = static_cast<int>(segment_count);
  const c10::cuda::CUDAGuard guard(query.device());
  auto output = at::empty_like(query);
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  at::Tensor partials;
  float* partial_ptr = nullptr;
  if (segments > 1) {
    partials = at::empty({rows, segment_count, kDim + 2}, query.options().dtype(at::kFloat));
    partial_ptr = partials.data_ptr<float>();
  }
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(segments));
  paged_decode_batched_kernel<<<grid, 128, 0, stream>>>(
      query.data_ptr<c10::BFloat16>(), key.data_ptr<c10::BFloat16>(), value.data_ptr<c10::BFloat16>(),
      table.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), output.data_ptr<c10::BFloat16>(), partial_ptr,
      static_cast<int>(q_heads), static_cast<int>(kv_heads), static_cast<int>(q_heads / kv_heads),
      static_cast<int>(table.size(1)), segments, key.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (segments > 1) {
    merge_decode_kernel<<<static_cast<unsigned int>(rows), kDim, 0, stream>>>(
        partial_ptr, output.data_ptr<c10::BFloat16>(), segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
