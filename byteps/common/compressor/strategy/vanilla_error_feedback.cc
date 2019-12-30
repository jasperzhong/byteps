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

#include "compressor/strategy/vanilla_error_feedback.h"

#include "logging.h"

namespace byteps {
namespace common {
namespace compressor {

CompressorRegistry::Register reg(
    "vanilla_error_feedback",
    [](const CompressorParam& param) -> CompressorPtr {
      auto iter = param.find("compressor_type");
      if (iter == param.end()) {
        BPS_LOG(FATAL) << "Vanilla Error-feedback Compressor needs parameter "
                          "\"compressor_type\"";
        return nullptr;
      }
      auto& registry = CompressorRegistry::instance();
      auto compressor_ptr = registry.create(iter->second, param);
      return std::unique_ptr<VanillaErrorFeedbackCompressor>(
          new VanillaErrorFeedbackCompressor(std::move(compressor_ptr)));
    });

VanillaErrorFeedbackCompressor::VanillaErrorFeedbackCompressor(
    std::unique_ptr<BaseCompressor> compressor_ptr)
    : ErrorFeedback(std::move(compressor_ptr)) {}

TensorType VanillaErrorFeedbackCompressor::UpdateGradient(
    const TensorType& grad) {
  // TODO
}

void VanillaErrorFeedbackCompressor::UpdateError(const TensorType& grad) {
  // TODO
}
}
}
}