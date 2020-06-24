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

#include "multibit.h"

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register
    reg("multibit_compressor",
        [](const kwargs_t& kwargs, size_t size,
           int dtype) -> std::unique_ptr<Compressor> {
          auto iter = kwargs.find("compressor_k");
          if (iter == kwargs.end()) {
            BPS_LOG(WARNING)
                << "Multibit Compressor needs parameter \"compressor_k\"";
            return nullptr;
          }
          int k = std::stoi(iter->second);
          BPS_LOG(DEBUG) << "Register Multibit Compressor "
                         << "k=" << k;
          return std::unique_ptr<Compressor>(new MultibitCompressor(size, k));
        });
}

void MultibitCompressor::Compress(tensor_t grad, tensor_t& compressed) {
  // TOOD
}

void MultibitCompressor::Decompress(tensor_t compressed,
                                    tensor_t& decompressed) {
  // TODO
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps