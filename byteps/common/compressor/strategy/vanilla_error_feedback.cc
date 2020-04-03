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

#include "vanilla_error_feedback.h"

#include "../../logging.h"

#include <bitset>


namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "vanilla_error_feedback",
    [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      // register cpr
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("error_feedback_type");
      auto compressor_ptr = CompressorRegistry::Create(kwargs_clone);
      BPS_CHECK_NE(compressor_ptr, nullptr);

      BPS_LOG(DEBUG) << "with Error feedback";
      return std::unique_ptr<VanillaErrorFeedbackCompressor>(
          new VanillaErrorFeedbackCompressor(std::move(compressor_ptr)));
    });
}

VanillaErrorFeedbackCompressor::VanillaErrorFeedbackCompressor(
    std::unique_ptr<BaseCompressor> compressor_ptr)
    : ErrorFeedback(std::move(compressor_ptr)) {}

VanillaErrorFeedbackCompressor::~VanillaErrorFeedbackCompressor() = default;

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void VanillaErrorFeedbackCompressor::UpdateGradient(ByteBuf grad, int dtype) {
  int local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
#ifdef BYTEPS_ENABLE_CUDA
  this->_cpu_reducer->sum(grad.data, _dev_error, grad.data, grad.size,
                          static_cast<DataType>(dtype), 1.0 / local_size);
#else
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.data, grad.size,
                          static_cast<DataType>(dtype), 1.0 / local_size);
#endif
}
#else
// server version decompressor
void VanillaErrorFeedbackCompressor::UpdateGradient(ByteBuf grad, int dtype) {
  int num_workers = atoi(getenv("DMLC_NUM_WORKER"));
#ifdef BYTEPS_ENABLE_CUDA
  this->_cpu_reducer->sum(grad.data, _dev_error, grad.data, grad.size,
                          static_cast<DataType>(dtype), 1.0 / num_workers);
#else
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.data, grad.size,
                          static_cast<DataType>(dtype), 1.0 / num_workers);
#endif
}
#endif

void VanillaErrorFeedbackCompressor::UpdateError(ByteBuf corrected, int dtype,
                                                 ByteBuf compressed) {
#ifdef BYTEPS_ENABLE_CUDA
  ByteBuf decompressed{_dev_error, corrected.size};
  CUDA_CALL(cudaMemcpy(_error.get(), this->_compressor_ptr->_dev_buf, compressed.size, cudaMemcpyDeviceToHost));
  BPS_LOG(INFO) << "compressed2: " << std::bitset<32>(reinterpret_cast<int*>(_error.get())[0]);
  Decompress({_error.get(), compressed.size}, dtype, {_error.get(), decompressed.size});
  // Decompress({this->_compressor_ptr->_dev_buf, compressed.size}, dtype,
  //            decompressed);
  // CUDA_CALL(cudaMemcpy(_error.get(), _dev_error, corrected.size, cudaMemcpyDeviceToHost));
  auto pf = reinterpret_cast<float*>(_error.get());
  float sum = 0;
  for (int i = 0; i < corrected.size / 4; ++i) {
    sum += pf[i];
  }
  BPS_LOG(INFO) << "decompressed sum=" << sum << " len=" << corrected.size / 4;

  auto scale =
      *reinterpret_cast<float*>(compressed.data + (compressed.size - 4));
  this->_cpu_reducer->sum(_dev_error, corrected.data, _dev_error,
                          corrected.size, static_cast<DataType>(dtype),
                          -1.0 * scale);
#else
  ByteBuf decompressed{_error.get(), corrected.size};
  Decompress(compressed, dtype, decompressed);
  this->_cpu_reducer->sum(_error.get(), corrected.data, decompressed.data,
                          corrected.size, static_cast<DataType>(dtype), -1.0);
#endif
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps