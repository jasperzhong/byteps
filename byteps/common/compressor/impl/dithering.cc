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

#include <bitset>
#include <cmath>

#include "encoding.h"

#include "../compressor_registry.h"
#include "dithering.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "dithering_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      auto iter = kwargs.find("compressor_k");
      if (iter == kwargs.end()) {
        BPS_LOG(FATAL) << "Randomk Compressor needs parameter \"compressor_k\"";
      }
      int k = std::stoi(iter->second);
      BPS_LOG(DEBUG) << "Register Dithering Compressor "
                     << "k=" << k;

      auto iter2 = kwargs.find("seed");
      unsigned int seed = 0;
      if (iter2 != kwargs.end()) {
        seed = std::stoul(iter2->second);
        BPS_CHECK(seed != 0) << "seed should not be 0";
      }

      auto iter3 = kwargs.find("partition");
      int ptype_int = 0;
      if (iter3 != kwargs.end()) {
        ptype_int = std::stoi(iter3->second);
        BPS_CHECK(ptype_int != 0 && ptype_int != 1) << "ptype should be 0 or 1";
      }
      auto ptype = static_cast<DitheringCompressor::PartitionType>(ptype_int);

      return std::unique_ptr<Compressor>(
          new DitheringCompressor(size, dtype, k, seed, ptype));
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

  bm::encoder encoder(reinterpret_cast<unsigned char*>(dst), _size);
  bm::bit_out<bm::encoder> bout(encoder);
  int last_non_zero_pos = -1;
  if (_ptype == PartitionType::LINEAR) {
    for (int i = 0; i < len; ++i) {
      float x = std::abs(src[i]);
      float fp = (x / l2) * _s;
      int low = std::floor(fp);
      int ret = low + _rng.Bernoulli(fp - low);
      if (ret) {
        int diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        bout.gamma(diff);
        bout.put_bit(std::signbit(src[i]));
        bout.gamma(ret);
      }
    }
  } else if (_ptype == PartitionType::NATURAL) {
    const int scale = 1 << (_s - 1);
    for (int i = 0; i < len; ++i) {
      float x = std::abs(src[i]);
      float fp = (x / l2) * scale;
      int low = RoundNextPow2(std::ceil(fp)) << 1;
      int ret = low * (1 + _rng.Bernoulli((fp - low) / low));
      if (ret) {
        int diff = i - last_non_zero_pos;
        last_non_zero_pos = i;
        bout.gamma(diff);
        bout.put_bit(std::signbit(src[i]));
        bout.gamma(ret);
      }
    }
  }
  bout.flush();
  auto size = encoder.size();

  // l2
  float* p_scale = reinterpret_cast<float*>(&dst[size / sizeof(index_t)]);
  *p_scale = l2;

  return {dst, size + sizeof(float)};
}

tensor_t DitheringCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename index_t, typename scalar_t>
tensor_t DitheringCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                             size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  const size_t size = compressed_size - sizeof(float);

  auto* p_scale = reinterpret_cast<const float*>(&src[size / sizeof(index_t)]);
  const float scale = *p_scale;

  auto ptr = const_cast<index_t*>(src);
  if ((void*)dst == (void*)src) {
    ptr = reinterpret_cast<index_t*>(_buf.get());
    std::memcpy(ptr, src, compressed_size);
  }
  std::memset(dst, 0, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

  bm::decoder decoder(reinterpret_cast<const unsigned char*>(ptr));
  bm::bit_in<bm::decoder> bin(decoder);
  int last_non_zero_pos = -1;
  while (decoder.size() < size) {
    int diff = bin.gamma();
    int i = last_non_zero_pos + diff;
    last_non_zero_pos = i;
    int signbit = bin.get_bits(1);
    int x = bin.gamma();
    float num = x * scale / s;
    dst[i] = (1 - (signbit << 1)) * num;
  }

  return {dst, _size};
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

template <typename index_t, typename scalar_t>
void DitheringCompressor::FastUpdateErrorImpl(scalar_t* error,
                                              scalar_t* corrected,
                                              const index_t* compressed,
                                              size_t compressed_size) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  const size_t size = compressed_size - sizeof(float);

  auto* p_scale =
      reinterpret_cast<const float*>(compressed + size / sizeof(index_t));
  const float scale = *p_scale;

  std::memcpy(error, corrected, _size);

  unsigned int s = _s;
  if (_ptype == PartitionType::NATURAL) {
    s = 1 << (_s - 1);
  }

  bm::decoder decoder(reinterpret_cast<const unsigned char*>(compressed));
  bm::bit_in<bm::decoder> bin(decoder);
  int last_non_zero_pos = -1;
  while (decoder.size() < size) {
    auto diff = bin.gamma();
    int i = last_non_zero_pos + diff;
    last_non_zero_pos = i;
    int signbit = bin.get_bits(1);
    auto x = bin.gamma();
    float num = x * scale / s;
    error[i] -= (1 - (signbit << 1)) * num;
  }
}

void DitheringCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                          tensor_t compressed) {
  SWITCH_TO_FAST_UPDATE_ERROR_IMPL_SWITCH(_dtype, FastUpdateErrorImpl,
                                          error.data, corrected.data,
                                          compressed.data, compressed.size);
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps