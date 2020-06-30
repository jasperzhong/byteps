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

#include <cmath>

#include "../compressor_registry.h"
#include "dithering.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register
    reg("dithering_compressor",
        [](const kwargs_t& kwargs, size_t size,
           DataType dtype) -> std::unique_ptr<Compressor> {
          auto iter = kwargs.find("compressor_k");
          if (iter == kwargs.end()) {
            BPS_LOG(WARNING)
                << "Multibit Compressor needs parameter \"compressor_k\"";
            return nullptr;
          }
          int k = std::stoi(iter->second);
          BPS_LOG(DEBUG) << "Register Multibit Compressor "
                         << "k=" << k;
          return std::unique_ptr<Compressor>(
              new DitheringCompressor(size, dtype, k));
        });
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                           size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  // normalize
  double l2 = 0.0;
  for (int i = 0; i < len; ++i) {
    l2 += src[i] * src[i];
  }
  l2 = std::sqrt(l2);

  switch (_ptype) {
    case PartitionType::LINEAR: {
      for (int i = 0; i < len; ++i) {
        float fp = (src[i] / l2) * k;
        int low = std::floor(fp);
        int ret = _rng.Bernoulli(fp - low);
†      }
      break;
    }

    case PartitionType::NATURAL: {
      break;
    }
    default:
      BPS_CHECK(0) << "Unsupported partition type: " << _ptype;
  }
}

tensor_t DitheringCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

tensor_t DitheringCompressor::Decompress(tensor_t compressed) {
  // TODO
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps