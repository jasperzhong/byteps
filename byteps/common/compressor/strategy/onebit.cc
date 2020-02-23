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

#include <chrono>

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg("onebit", [](const kwargs_t& kwargs) {
  BPS_LOG(DEBUG) << "Register Onebit Compressor";
  return std::unique_ptr<BaseCompressor>(new OnebitCompressor());
});
}

OnebitCompressor::OnebitCompressor() = default;

OnebitCompressor::~OnebitCompressor() = default;

size_t Packing(void* dst, void* src, size_t len) {
  constexpr int MASK = 1;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  size_t chunk_size = (len + padding_len) / PACKING_SIZE;

  auto ptr_dst = reinterpret_cast<int*>(dst);
  auto ptr_src = reinterpret_cast<int*>(src);
#pragma unroll
  for (int i = 0; i < PACKING_SIZE; ++i) {
    for (int j = 0; j < chunk_size; ++j) {
      ptr_dst[j] <<= 1;
      ptr_dst[j] |= ptr_src[i * chunk_size + j] & MASK;
    }
  }

  return chunk_size;
}

void OnebitCompressor::Compress(ByteBuf grad, int dtype, ByteBuf* compressed) {
  BPS_CHECK(compressed);
  auto reduced_len = _cpu_reducer->sign(grad.data, grad.data, grad.size,
                                        static_cast<DataType>(dtype));

  auto compressed_size = Packing(_buf.get(), grad.data, reduced_len);

  compressed->data = _buf.get();
  compressed->size = compressed_size;
}

void Unpacking(void* dst, void* src, size_t size) {
  constexpr int MASK = 1;
  auto chunk_size = size / sizeof(int);

  auto ptr_dst = reinterpret_cast<int*>(dst);
  auto ptr_src = reinterpret_cast<int*>(src);
#pragma unroll
  for (int i = PACKING_SIZE - 1; i >= 0; --i) {
    for (int j = 0; j < chunk_size; ++j) {
      ptr_dst[i * chunk_size + j] = -(((ptr_src[j] & MASK) << 1) - 1);
      ptr_src[j] >>= 1;
    }
  }
}

#ifndef BYTEPS_BUILDING_SERVER
// worker version decompressor
void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf* decompressed) {
  BPS_CHECK(decompressed);
  Unpacking(_buf.get(), compressed.data, compressed.size);
  _cpu_reducer->int2fp(compressed.data, _buf.get(), decompressed->size,
                       static_cast<DataType>(dtype));
}

#else

void OnebitCompressor::Decompress(ByteBuf compressed, int dtype,
                                  ByteBuf* decompressed) {
  BPS_CHECK(decompressed);
  Unpacking(_buf.get(), compressed.data, compressed.size);
  _cpu_reducer->int2fp(compressed.data, _buf.get(), decompressed->size,
                       static_cast<DataType>(dtype));
  decompressed->data = _buf.get();
}
#endif

}  // namespace compressor
}  // namespace common
}  // namespace byteps