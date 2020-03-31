// Copyright 2019 Amazon Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {

ErrorFeedback::ErrorFeedback(std::unique_ptr<BaseCompressor> compressor_ptr)
    : _compressor_ptr(std::move(compressor_ptr)) {}

ErrorFeedback::~ErrorFeedback() {
#ifdef BYTEPS_ENABLE_CUDA
  cudaFree(_dev_error);
#endif
};

void ErrorFeedback::Init(size_t aligned_size) {
  BaseCompressor::Init(aligned_size);
  _compressor_ptr->Init(aligned_size);
  _error.reset(new char[aligned_size]);
  memset(_error.get(), 0, aligned_size);
#ifdef BYTEPS_ENABLE_CUDA
  cudaMalloc(&_dev_error, aligned_size);
  cudaMemset(_dev_error, 0, aligned_size);
#endif
}

void ErrorFeedback::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  auto corrected = grad;
#ifdef BYTEPS_ENABLE_CUDA
  CUDA_CALL(cudaMemcpyAsync(_dev_buf, grad.data, grad.size,
                            cudaMemcpyHostToDevice, _stream));
  corrected = {_dev_buf, grad.size};
#endif
  // before: grad += error
  UpdateGradient(corrected, dtype);
#ifdef BYTEPS_ENABLE_CUDA
  CUDA_CALL(cudaStreamSynchronize(_stream));
#endif
  // compress
  _compressor_ptr->Compress(corrected, dtype, compressed);

  // UpdateError(corrected, dtype, compressed);
#ifdef BYTEPS_ENABLE_CUDA
  CUDA_CALL(cudaStreamSynchronize(_stream));
#endif
}

void ErrorFeedback::Decompress(ByteBuf compressed, int dtype,
                               ByteBuf& decompressed) {
  _compressor_ptr->Decompress(compressed, dtype, decompressed);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps