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

#include "onebit.h"

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "onebit_compressor", [](const kwargs_t& kwargs) {
      BPS_LOG(DEBUG) << "Register Onebit Compressor";
      bool scaled = false;
      auto iter = kwargs.find("compressor_onebit_scaling");
      if (iter != kwargs.end()) {
        if (iter->second == "true" || iter->second == "True") scaled = true;
      }
      if (scaled) {
        return std::unique_ptr<BaseCompressor>(new OnebitCompressor(true));
      }
      return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
    });
}

OnebitCompressor::OnebitCompressor(bool use_scale) : _use_scale(use_scale){};

OnebitCompressor::~OnebitCompressor() = default;

template <typename index_t, typename scalar_t>
size_t OnebitCompressor::PackingImpl(index_t* dst, const scalar_t* src,
                                     size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * sizeof(char);

  float scale = 1.0f;
  if (_use_scale) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
      dst[i] = src[i] < 0;
      sum += abs(src[i]);
    }
    scale = sum / len;
  } else {
    for (size_t i = 0; i < len; ++i) {
      dst[i] = src[i] < 0;
    }
  }

  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      dst[j] <<= 1;
      dst[j] |= dst[i * chunk_size + j] & 0x01;
    }
  }

  float* p_scale = reinterpret_cast<float*>(&dst[chunk_size]);
  *p_scale = scale;

  return chunk_size * sizeof(index_t) + sizeof(float);
}

size_t OnebitCompressor::Packing(void* dst, const void* src, size_t len,
                                 int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
      return PackingImpl(reinterpret_cast<int8_t*>(dst),
                         reinterpret_cast<const int8_t*>(src),
                         len / sizeof(int8_t));
    case BYTEPS_UINT8:
      return PackingImpl(reinterpret_cast<int8_t*>(dst),
                         reinterpret_cast<const uint8_t*>(src),
                         len / sizeof(uint8_t));
    // case BYTEPS_FLOAT16:
    //   return PackingImpl(reinterpret_cast<int16_t*>(dst), len);
    case BYTEPS_INT32:
      return PackingImpl(reinterpret_cast<int32_t*>(dst),
                         reinterpret_cast<const int32_t*>(src),
                         len / sizeof(int32_t));
    case BYTEPS_FLOAT32:
      return PackingImpl(reinterpret_cast<int32_t*>(dst),
                         reinterpret_cast<const float*>(src),
                         len / sizeof(int32_t));
    case BYTEPS_INT64:
      return PackingImpl(reinterpret_cast<int64_t*>(dst),
                         reinterpret_cast<const int64_t*>(src),
                         len / sizeof(int64_t));
    case BYTEPS_FLOAT64:
      return PackingImpl(reinterpret_cast<int64_t*>(dst),
                         reinterpret_cast<const double*>(src),
                         len / sizeof(int64_t));
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

void OnebitCompressor::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  compressed.size = Packing(_buf.get(), grad.data, grad.size, dtype);
  compressed.data = _buf.get();
}

template <typename scalar_t, typename index_t>
size_t OnebitCompressor::UnpackingImpl(scalar_t* dst, const index_t* src,
                                       size_t size) {
  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(index_t) * sizeof(char);
  size_t chunk_size = (size - sizeof(float)) / sizeof(index_t);

  float* pf = reinterpret_cast<const float*>(src + chunk_size);
  float scale = *pf;

  unsigned int mask = 1;
  // scale = 1, no need to scale 
  if (abs(scale - 1) < 1e-6) {
    for (int i = PACKING_SIZE - 1; i >= 0; --i) {
      for (int j = 0; j < chunk_size; ++j) {
        int sign_bit = (src[j] & mask) >> (PACKING_SIZE - i - 1);
        int sign = -((sign_bit << 1) - 1);
        dst[i * chunk_size + j] = sign;
      }
      mask <<= 1;
    }
  } else {
    for (int i = PACKING_SIZE - 1; i >= 0; --i) {
      for (int j = 0; j < chunk_size; ++j) {
        int sign_bit = (src[j] & mask) >> (PACKING_SIZE - i - 1);
        int sign = -((sign_bit << 1) - 1);
        dst[i * chunk_size + j] = sign * scale;
      }
      mask <<= 1;
    }
  }
}

size_t OnebitCompressor::Unpacking(void* dst, const void* src, size_t len,
                                   int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
      return UnpackingImpl(reinterpret_cast<int8_t*>(dst),
                           reinterpret_cast<const int8_t*>(src), len);
    case BYTEPS_UINT8:
      return UnpackingImpl(reinterpret_cast<uint8_t*>(dst),
                           reinterpret_cast<const int8_t*>(src), len);
    // TODO:
    // case BYTEPS_FLOAT16:
    //   return UnpackingImpl(reinterpret_cast<uint16_t*>(dst),
    //                     reinterpret_cast<const int16_t*>(src), len);
    case BYTEPS_INT32:
      return UnpackingImpl(reinterpret_cast<int32_t*>(dst),
                           reinterpret_cast<const int32_t*>(src), len);
    case BYTEPS_FLOAT32:
      return UnpackingImpl(reinterpret_cast<float*>(dst),
                           reinterpret_cast<const int32_t*>(src), len);
    case BYTEPS_INT64:
      return UnpackingImpl(reinterpret_cast<int64_t*>(dst),
                           reinterpret_cast<const int64_t*>(src), len);
    case BYTEPS_FLOAT64:
      return UnpackingImpl(reinterpret_cast<double*>(dst),
                           reinterpret_cast<const int64_t*>(src), len);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  BPS_CHECK(decompressed.data);
  Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
}

#else

void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf& decompressed) {
  if (decompressed.data == nullptr) decompressed.data = _buf.get();
  Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps