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

#include <bitset>

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "onebit_compressor", [](const kwargs_t& kwargs) {
      BPS_LOG(DEBUG) << "Register Onebit Compressor";
      if (kwargs.find("compressor_onebit_enable_scale") != kwargs.end()) {
        return std::unique_ptr<BaseCompressor>(new OnebitCompressor(true));
      }
      return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
    });
}

OnebitCompressor::OnebitCompressor(bool use_scale) : _use_scale(use_scale){};

OnebitCompressor::~OnebitCompressor() = default;

template <typename T>
size_t OnebitCompressor::_Packing(T* data, size_t len) {
  constexpr int PACKING_SIZE = sizeof(T) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  for (int i = 1; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      data[j] <<= 1;
      data[j] |= data[i * chunk_size + j] & 0x01;
    }
  }

  return chunk_size * sizeof(T);
}

size_t OnebitCompressor::Packing(void* data, size_t len, int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
    case BYTEPS_UINT8:
      return _Packing(reinterpret_cast<int8_t*>(data), len);
    case BYTEPS_FLOAT16:
      return _Packing(reinterpret_cast<int16_t*>(data), len);
    case BYTEPS_INT32:
    case BYTEPS_FLOAT32:
      return _Packing(reinterpret_cast<int32_t*>(data), len);
    case BYTEPS_INT64:
    case BYTEPS_FLOAT64:
      return _Packing(reinterpret_cast<int64_t*>(data), len);
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

void OnebitCompressor::Compress(ByteBuf grad, int dtype, ByteBuf& compressed) {
  float scale = 1.0;
  float norm1 = 1.0;

  if (_use_scale) {
#ifdef BYTEPS_ENABLE_CUDA
    norm1 = _cpu_reducer->norm1(grad.data, _dev_buf, _buf.get(), grad.size,
                                static_cast<DataType>(dtype));
#else
    _cpu_reducer->norm1(grad.data, &norm1, grad.size,
                        static_cast<DataType>(dtype));
#endif
  }

#ifdef BYTEPS_ENABLE_CUDA
  auto reduced_len = _cpu_reducer->sign(_dev_buf, grad.data, grad.size,
                                        static_cast<DataType>(dtype));
#else
  auto reduced_len = _cpu_reducer->sign(_buf.get(), grad.data, grad.size,
                                        static_cast<DataType>(dtype));
#endif

#ifdef BYTEPS_ENABLE_CUDA
  auto compressed_size = PackingCuda(_dev_buf, reduced_len, dtype);
  CUDA_CALL(cudaMemcpy(_buf.get(), _dev_buf, compressed_size,
                       cudaMemcpyDeviceToHost));
#else
  auto compressed_size = Packing(_buf.get(), reduced_len, dtype);
#endif
  BPS_LOG(INFO) << "compressed: " << std::bitset<32>(reinterpret_cast<int*>(_buf.get())[0]);
  scale = norm1 / (grad.size / getDataTypeLength(dtype));
  auto pf = reinterpret_cast<float*>(_buf.get() + compressed_size);
  *pf = scale;

  compressed.data = _buf.get();
  compressed.size = compressed_size + sizeof(float);
}

template <typename T1, typename T2>
size_t OnebitCompressor::_Unpacking(T1* dst, const T2* src, size_t size) {
  static_assert(sizeof(T1) == sizeof(T2), "T1 should be the same size as T2");
  constexpr int PACKING_SIZE = sizeof(T2) * 8;
  auto chunk_size = (size - sizeof(float)) / sizeof(T2);

  float scale;
  auto pf = reinterpret_cast<const float*>(src + chunk_size);
  scale = *pf;

  unsigned int mask = 1;
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    for (int j = 0; j < chunk_size; ++j) {
      int sign_bit = (src[j] & mask) >> (PACKING_SIZE - i - 1);
      int sign = -((sign_bit << 1) - 1);
      dst[i * chunk_size + j] = sign * scale;
    }
    mask <<= 1;
  }

  return chunk_size;
}

size_t OnebitCompressor::Unpacking(void* dst, const void* src, size_t len,
                                   int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
      return _Unpacking(reinterpret_cast<int8_t*>(dst),
                        reinterpret_cast<const int8_t*>(src), len);
    case BYTEPS_UINT8:
      return _Unpacking(reinterpret_cast<uint8_t*>(dst),
                        reinterpret_cast<const int8_t*>(src), len);
    // TODO:
    // case BYTEPS_FLOAT16:
    //   return _Unpacking(reinterpret_cast<uint16_t*>(dst),
    //                     reinterpret_cast<const int16_t*>(src), len);
    case BYTEPS_INT32:
      return _Unpacking(reinterpret_cast<int32_t*>(dst),
                        reinterpret_cast<const int32_t*>(src), len);
    case BYTEPS_FLOAT32:
      return _Unpacking(reinterpret_cast<float*>(dst),
                        reinterpret_cast<const int32_t*>(src), len);
    case BYTEPS_INT64:
      return _Unpacking(reinterpret_cast<int64_t*>(dst),
                        reinterpret_cast<const int64_t*>(src), len);
    case BYTEPS_FLOAT64:
      return _Unpacking(reinterpret_cast<double*>(dst),
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
#ifdef BYTEPS_ENABLE_CUDA
  if (compressed.data == decompressed.data) {
    Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
  } else {
    Unpacking(decompressed.data, compressed.data, compressed.size, dtype);

    // UnpackingCuda(decompressed.data, compressed.data, compressed.size, dtype);
  }
#else
  Unpacking(decompressed.data, compressed.data, compressed.size, dtype);
#endif
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
