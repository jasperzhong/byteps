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
#pragma omp parallel for simd num_threads(4) reduction(+ : l2)
  for (int i = 0; i < len; ++i) {
    l2 += src[i] * src[i];
  }
  l2 = std::sqrt(l2);

  BitWriter bit_writer(dst);
  int last_non_zero_pos = 0;
  if (_ptype == PartitionType::LINEAR) {
    for (int i = 0; i < len; ++i) {
      int x = std::abs(src[i]);
      float fp = (x / l2) * _s;
      int low = std::floor(fp);
      int ret = low + _rng.Bernoulli(fp - low);
      if (ret) {
        int diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        bit_writer.Put(std::signbit(src[i]));
        EliasDeltaEncode(bit_writer, ret);
      }
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const int scale = 1 << (_s - 1);
    for (int i = 0; i < len; ++i) {
      int x = std::abs(src[i]);
      float fp = (x / l2) * scale;
      int low = RoundNextPow2(std::ceil(fp)) << 1;
      int ret = low * (1 + _rng.Bernoulli((fp - low) / low));
      if (ret) {
        int diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        EliasDeltaEncode(bit_writer, diff);
        bit_writer.Put(std::signbit(src[i]));
        EliasDeltaEncßode(bit_writer, ret);
      }
    }
  }

  bit_writer.Pad();

  float* p_scale = reinterpret_cast<float*>(&dst[bit_writer.ints()]);
  *p_scale = l2;

  return {dst, bit_writer.ints() * sizeof(index_t) + sizeof(float)};
}

tensor_t DitheringCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                             size_t compressed_size) {
  
}

tensor_t DitheringCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps