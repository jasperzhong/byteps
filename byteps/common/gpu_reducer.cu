#include "cpu_reducer.h"
#include "logging.h"

__global__ void sum_kernel(float* dst, const float* src, size_t len,
                           float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] += src[i] * alpha;
}

__global__ void sum_kernel(float* dst, const float* src1, const float* src2,
                           size_t len, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] = src1[i] + src2[i] * alpha;
}

__global__ void sign_kernel(int* dst, const float* src, size_t len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] = signbit(src[i]);
}

__global__ void norm1_kernel(const float* src, float* out, size_t len) {
  // max size 16KB
  __shared__ float vec[1024];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  vec[tid] = (idx < len) ? src[idx] : 0;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      vec[tid] = abs(vec[tid]) + abs(vec[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, vec[0]);
}

namespace byteps {
namespace common {
constexpr int BLOCK_PER_GRID = 1024;
// int CpuReducer::sum(void* dev_dst, const void* dev_src, size_t len, int
// dtype,
//                     float alpha) {
//   int thread_per_block = ((len/4) + BLOCK_PER_GRID) / BLOCK_PER_GRID;
//   sum_kernel<<<BLOCK_PER_GRID, thread_per_block, 0, *_stream>>>(
//       reinterpret_cast<float*>(dev_dst),
//       reinterpret_cast<const float*>(const_cast<void*>(dev_src)), len / 4,
//       alpha);
//   return 0;
// }

int CpuReducer::sum(void* dev_dst, const void* dev_src1, const void* dev_src2,
                    size_t len, int dtype, float alpha) {
  int thread_per_block = ((len / 4) + BLOCK_PER_GRID) / BLOCK_PER_GRID;
  sum_kernel<<<BLOCK_PER_GRID, thread_per_block, 0, *_stream>>>(
      reinterpret_cast<float*>(dev_dst),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src1)),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src2)), len / 4,
      alpha);
  return 0;
}

int CpuReducer::sign(void* dev_dst, const void* dev_src, size_t len,
                     int dtype) {
  int thread_per_block = ((len / 4) + BLOCK_PER_GRID) / BLOCK_PER_GRID;
  sign_kernel<<<BLOCK_PER_GRID, thread_per_block, 0, *_stream>>>(
      reinterpret_cast<int*>(dev_dst),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src)), len / 4);
  return len / 4;
}

int CpuReducer::norm1(const void* dev_src, float* dev_out, size_t len,
                      int dtype) {
  int x = ((len / 4) + BLOCK_PER_GRID) / BLOCK_PER_GRID;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  ++x;
  norm1_kernel<<<BLOCK_PER_GRID, x, 0, *_stream>>>(
      reinterpret_cast<const float*>(const_cast<void*>(dev_src)), dev_out,
      len / 4);
  return 0;
}

}  // namespace common
}  // namespace byteps
