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

#include <queue>

#include "../compressor_registry.h"
#include "topk.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "topk_compressor",
    [](const kwargs_t& kwargs, size_t size,
       int dtype) -> std::unique_ptr<Compressor> {
      auto iter = kwargs.find("compressor_k");
      if (iter == kwargs.end()) {
        BPS_LOG(FATAL) << "Topk Compressor needs parameter \"compressor_k\"";
      }
      int k = std::stoi(iter->second);
      BPS_LOG(DEBUG) << "Register Topk Compressor "
                     << "k=" << k;
      return std::unique_ptr<Compressor>(new TopkCompressor(size, k));
    });
}

template <typename index_t, typename scalar_t>
size_t TopkCompressor::PackingImpl(index_t* dst, const scalar_t* src,
                                   size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  BPS_CHECK_LE(this->_k, len / 2);
  using pair_t = std::pair<index_t, scalar_t>;
  using container_t = std::vector<pair_t>;
  auto comp = [](const pair_t& lhs, const pair_t& rhs) {
    return lhs.second > rhs.second;
  };

  auto beg = reinterpret_cast<pair_t*>(dst);
  size_t size = 0;
  for (index_t i = 0; i < len; ++i) {
    if (i < this->_k) {
      beg[size] = std::make_pair(i, src[i]);
      size++;
      std::push_heap(beg, beg + size, comp);
    } else {
      auto& top = *beg;
      // note: compare absolute value
      if (std::abs(src[i]) > std::abs(top.second)) {
        std::pop_heap(beg, beg + size, comp);
        beg[size - 1] = std::make_pair(i, src[i]);
        std::push_heap(beg, beg + size, comp);
      }
    }
  }

  return this->_k * sizeof(pair_t);
}

size_t TopkCompressor::Packing(const void* src, size_t size, int dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return PackingImpl(reinterpret_cast<int32_t*>(_buf.get()),
                         reinterpret_cast<const float*>(src),
                         size / sizeof(int32_t));
    case BYTEPS_FLOAT64:
      return PackingImpl(reinterpret_cast<int64_t*>(_buf.get()),
                         reinterpret_cast<const double*>(src),
                         size / sizeof(int64_t));
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 0;
}

tensor_t TopkCompressor::Compress(tensor_t grad) {
  auto compressed_size = Packing(grad.data, grad.size, grad.dtype);
  return {_buf.get(), compressed_size};
}

template <typename index_t, typename scalar_t>
void TopkCompressor::UnpackingImpl(scalar_t* dst, const index_t* src,
                                   size_t len, size_t src_len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<index_t, scalar_t>;
  auto ptr = reinterpret_cast<const pair_t*>(src);

  if ((void*)dst == (void*)src) {
    auto buf = reinterpret_cast<pair_t*>(_buf.get());
    std::copy(ptr, ptr + len, buf);
    ptr = const_cast<const pair_t*>(buf);
  }

  // reset to zeros
  std::fill(dst, dst + src_len, 0);
  for (auto i = 0; i < len; ++i) {
    auto& pair = ptr[i];
    dst[pair.first] = pair.second;
  }
}

void TopkCompressor::Unpacking(void* dst, const void* src, size_t size,
                               size_t src_size, int dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return UnpackingImpl(reinterpret_cast<float*>(dst),
                           reinterpret_cast<const int32_t*>(src),
                           size / sizeof(float) / 2, src_size / sizeof(float));
    case BYTEPS_FLOAT64:
      return UnpackingImpl(
          reinterpret_cast<double*>(dst), reinterpret_cast<const int64_t*>(src),
          size / sizeof(double) / 2, src_size / sizeof(double));
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
}

tensor_t TopkCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst_ptr = _buf.get();
#else
  auto dst_ptr = compressed.data;
#endif
  Unpacking(dst_ptr, compressed.data, compressed.size, _size, compressed.dtype);
  return {dst_ptr, _size};
}

template <typename index_t, typename scalar_t>
void TopkCompressor::FastUpdateErrorImpl(scalar_t* error,
                                         const index_t* compressed,
                                         size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");
  using pair_t = std::pair<index_t, scalar_t>;

  auto ptr = reinterpret_cast<const pair_t*>(compressed);
  for (auto i = 0; i < this->_k; ++i) {
    auto& pair = ptr[i];
    error[pair.first] = 0;
  }
}

void TopkCompressor::FastUpdateError(tensor_t error, tensor_t corrected,
                                     tensor_t compressed) {
  std::memcpy(error.data, corrected.data, corrected.size);
  switch (corrected.dtype) {
    case BYTEPS_FLOAT32:
      return FastUpdateErrorImpl(
          reinterpret_cast<float*>(error.data),
          reinterpret_cast<const int32_t*>(compressed.data),
          corrected.size / sizeof(float));
    case BYTEPS_FLOAT64:
      return FastUpdateErrorImpl(
          reinterpret_cast<double*>(error.data),
          reinterpret_cast<const int64_t*>(compressed.data),
          corrected.size / sizeof(double));
    default:
      BPS_CHECK(0) << "Unsupported data type: " << corrected.dtype;
  }
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps