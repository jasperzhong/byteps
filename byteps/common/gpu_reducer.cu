#include <iostream>

#include "compressor/strategy/onebit.h"
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

__global__ void norm1_kernel(const float* src, float* dst, size_t len) {
  // max size 16KB
  __shared__ float vec[1024];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  vec[tid] = (idx < len) ? src[idx] : 0;
  __syncthreads();

  for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride) {
      vec[tid] = abs(vec[tid]) + abs(vec[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) dst[blockIdx.x] = vec[0];
}

constexpr int PACKING_SIZE = 32;
__global__ void packing(int* data, size_t chunk_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunk_size) return;
#pragma unroll
  for (int i = 1; i < PACKING_SIZE; ++i) {
    data[idx] <<= 1;
    data[idx] |= data[i * chunk_size + idx] & 0x01;
  }
}

__global__ void unpacking(float* dst, const int* src, size_t chunk_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunk_size) return;
  unsigned int mask = 1;
#pragma unroll
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    int sign_bit = (src[idx] & mask) >> (PACKING_SIZE - i - 1);
    int sign = -((sign_bit << 1) - 1);
    dst[i * chunk_size + idx] = sign;
    mask <<= 1;
  }
}

namespace byteps {
namespace common {
constexpr int BLOCK_PER_GRID = 1024;
// int CpuReducer::sum(void* dev_dst, const void* dev_src, size_t len, int
// dtype,
//                     float alpha) {
//   int thread_per_block = ((len/4) + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
//   sum_kernel<<<BLOCK_PER_GRID, thread_per_block>>>(
//       reinterpret_cast<float*>(dev_dst),
//       reinterpret_cast<const float*>(const_cast<void*>(dev_src)), len / 4,
//       alpha);
//   return 0;
// }

int CpuReducer::sum(void* dev_dst, const void* dev_src1, const void* dev_src2,
                    size_t len, int dtype, float alpha) {
  int thread_per_block = ((len / 4) + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
  sum_kernel<<<BLOCK_PER_GRID, thread_per_block>>>(
      reinterpret_cast<float*>(dev_dst),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src1)),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src2)), len / 4,
      alpha);
  return 0;
}

int CpuReducer::sign(void* dev_dst, const void* dev_src, size_t len,
                     int dtype) {
  int thread_per_block = ((len / 4) + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
  sign_kernel<<<BLOCK_PER_GRID, thread_per_block>>>(
      reinterpret_cast<int*>(dev_dst),
      reinterpret_cast<const float*>(const_cast<void*>(dev_src)), len / 4);
  return len / 4;
}

float CpuReducer::norm1(const void* dev_src, void* dev_dst, void* dst,
                        size_t len, int dtype) {
  int x = ((len / 4) + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  ++x;

  norm1_kernel<<<BLOCK_PER_GRID, x>>>(
      reinterpret_cast<const float*>(const_cast<void*>(dev_src)),
      reinterpret_cast<float*>(dev_dst), len / 4);

  cudaMemcpy(dst, dev_dst, BLOCK_PER_GRID * 4, cudaMemcpyDeviceToHost);

  float ret = 0;
  auto p_dst = reinterpret_cast<float*>(dst);
  for (int i = 0; i < BLOCK_PER_GRID; ++i) {
    ret += p_dst[i];
  }

  return ret;
}

namespace compressor {

size_t OnebitCompressor::PackingCuda(void* data, size_t len, int dtype) {
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  int thread_per_block = (chunk_size + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
  packing<<<BLOCK_PER_GRID, thread_per_block>>>(reinterpret_cast<int*>(data),
                                                chunk_size);
  return chunk_size * 4;
}

size_t OnebitCompressor::UnpackingCuda(void* dst, const void* src, size_t len,
                                       int dtype) {
  auto chunk_size = (len - sizeof(float)) / 4;
  int thread_per_block = (chunk_size + BLOCK_PER_GRID - 1) / BLOCK_PER_GRID;
  unpacking<<<BLOCK_PER_GRID, thread_per_block>>>(
      reinterpret_cast<float*>(dst),
      reinterpret_cast<const int*>(const_cast<void*>(src)), chunk_size);
  return chunk_size;
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps
