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

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "vanilla_ef",
    [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      // register cpr
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("ef_type");
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

VanillaErrorFeedbackCompressor::~VanillaErrorFeedbackCompressor() {
  munmap(_mm, 8);
  close(_fd);
}

void VanillaErrorFeedbackCompressor::Init(size_t aligned_size) {
  ErrorFeedback::Init(aligned_size);
  _fd = open("lr.s", O_RDONLY);
  BPS_CHECK(_fd > 0) << "open lr.s failed, errno=" << strerror(errno);
  void* ptr = mmap(0, 8, PROT_READ, MAP_SHARED, _fd, 0);
  BPS_CHECK_NE(ptr, MAP_FAILED) << "mmap failed, errno=" << strerror(errno);
  _mm = ptr;
  _pre_lr = _cur_lr = *reinterpret_cast<double*>(_mm);
}

void VanillaErrorFeedbackCompressor::UpdateGradient(ByteBuf grad, int dtype) {
  _cur_lr = *reinterpret_cast<double*>(_mm);
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.size,
                          static_cast<DataType>(dtype), (_pre_lr / _cur_lr));
  _pre_lr = _cur_lr;
}

void VanillaErrorFeedbackCompressor::UpdateError(ByteBuf corrected, int dtype,
                                                 ByteBuf compressed) {
  // ByteBuf decompressed{_error.get(), corrected.size};
  // Decompress(compressed, dtype, decompressed);
  float scale = *reinterpret_cast<float*>(compressed.data + compressed.size -
                                          sizeof(float));
  size_t len = corrected.size / getDataTypeLength(dyte);
  auto err_fp_ptr = reinterpret_cast<float*>(_error.get());
  auto err_int_ptr = reinterpret_cast<int32_t*>(_error.get());
  auto p_ptr = reinterpret_cast<int32_t*>(corrected.data);
  for (size_t i = 0; i < len; ++i) {
    err_fp_ptr[i] = p_ptr[i] - scale * err_int_ptr[i];
  }
  // this->_cpu_reducer->sum(_error.get(), corrected.data, _error.get(),
  //                         corrected.size, static_cast<DataType>(dtype), -scale);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps