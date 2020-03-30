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

__global__ void norm1_kernel(float* src, float* out, size_t len) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) return;

  float* data = src + blockIdx.x * blockDim.x;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      data[tid] = abs(data[tid]) + abs(data[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, data[0]);
}

namespace byteps {
namespace common {

int CpuReducer::sum(void* dev_dst, const void* dev_src, size_t len, int dtype,
                    float alpha) {
  sum_kernel<<<_block_per_grid, _thread_per_block, 0, *_stream>>>(
      reinterpret_cast<float*>(dev_dst),
      reinterpret_cast<float*>(const_cast<void*>(dev_src)), len / 4, alpha);
  return 0;
}

int CpuReducer::sum(void* dev_dst, const void* dev_src1, const void* dev_src2,
                    size_t len, int dtype, float alpha) {
  sum_kernel<<<_block_per_grid, _thread_per_block, 0, *_stream>>>(
      reinterpret_cast<float*>(dev_dst),
      reinterpret_cast<float*>(const_cast<void*>(dev_src1)),
      reinterpret_cast<float*>(const_cast<void*>(dev_src2)), len / 4, alpha);
  return 0;
}

int CpuReducer::sign(void* dev_dst, const void* dev_src, size_t len,
                     int dtype) {
  sign_kernel<<<_block_per_grid, _thread_per_block, 0, *_stream>>>(
      reinterpret_cast<int*>(dev_dst),
      reinterpret_cast<float*>(const_cast<void*>(dev_src)), len / 4);
  return len / 4;
}

int CpuReducer::norm1(void* dev_src, float* dev_out, size_t len, int dtype) {
  norm1_kernel<<<_block_per_grid, _thread_per_block, 0, *_stream>>>(
      reinterpret_cast<float*>(dev_src), dev_out, len / 4);

  return 0;
}

}  // namespace common
}  // namespace byteps
