#include "cpu_reducer"

namespace byteps {
namespace common {

int CpuReducer::sum(void* dst, const void* src, size_t len,
                               int dtype, float alpha) {
  cudaMemcpy(dev_src1, src, len, cudaMemcpyHostToDevice);

  _sum<<<_block_per_grid, _thread_per_block>>>(dev_dst, dev_src1, len / 4,
                                               alpha);

  cudaMemcpy(dst, dev_dst, len, cudaMemcpyDeviceToHost);
  return 0;
}

int CpuReducer::sum(void* dst, const void* src1, const void* src2,
                               size_t len, int dtype, float alpha) {
  cudaMemcpy(dev_src1, src1, len, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_src2, src2, len, cudaMemcpyHostToDevice);

  _sum<<<_block_per_grid, _thread_per_block>>>(dev_dst, dev_src1, dev_src2,
                                               len / 4, alpha);

  cudaMemcpy(dst, dev_dst, len, cudaMemcpyDeviceToHost);
  return 0;
}

int CpuReducer::sign(void* dst, const void* src, size_t len,
                                int dtype) {
  cudaMemcpy(dev_src1, src, len, cudaMemcpyHostToDevice);

  _sign<<<_block_per_grid, _thread_per_block>>>(dev_dst, dev_src1, len / 4);

  cudaMemcpy(dst, dev_dst, len, cudaMemcpyDeviceToHost);
  return 0;
}

float CpuReducer::norm1(const void* src, size_t len,
                                   int dtype) {
  cudaMemcpy(dev_src1, src, len, cudaMemcpyHostToDevice);

  _norm1<<<_block_per_grid, _thread_per_block>>>(dev_src1, dev_scalar, len / 4);

  float ret = 0.0;
  cudaMemcpy(&ret, dev_scalar, 4, cudaMemcpyDeviceToHost);
  return ret;
}

__global__ void _sum(float* dst, const float* src, size_t len, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] += src[i] * alpha;
}

__global__ void _sum(float* dst, const float* src1, const float* src2,
                     size_t len, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] = src1[i] + src2[i] * alpha;
}

__global__ void _sign(float* dst, const float* src, size_t len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) dst[i] = signbit(src[i]);
}

__global__ void _norm1(const float* src, float* out, size_t len) {
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

}  // namespace common
}  // namespace byteps
