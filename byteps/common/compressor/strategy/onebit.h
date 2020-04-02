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

#ifndef BYTEPS_COMPRESS_STRAT_ONEBIT_H
#define BYTEPS_COMPRESS_STRAT_ONEBIT_H

#include "../base_compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Onebit Compressor
 *
 * paper: SIGNSGD: Compressed Optimisation for Non-Convex Problems
 * https://arxiv.org/pdf/1802.04434.pdf
 *
 * each worker i:
 *    c_i <- sign(grad)
 *
 * server: majority vote
 *    sign(\sum_i c_i)
 *
 * \note this is a deterministic algorithm.
 * \note 0 represents positive and 1 represents negative.
 */
class OnebitCompressor : public BaseCompressor {
 public:
  OnebitCompressor(bool use_scale = false);
  virtual ~OnebitCompressor();
  /*!
   * \brief Allocate encoding buffer for compression.
   * \param aligned_size aligned size
   */
  void Init(size_t aligned_size);
  /*!
   * \brief Compress function
   *
   * compress and pack into byte array.
   * each bit represents a sign.
   *
   * \param grad gradient tensor
   * \param dtype data type
   * \param compressed compressed tensor
   */
  void Compress(ByteBuf grad, int dtype, ByteBuf& compressed) override;

  /*!
   * \brief Decompress
   *
   * unpack from byte array to FP tensor
   *
   * \param compressed compressed tensor
   * \param dtype data type
   * \param src_size uncompressed tensor size
   * \param decompressed decompressed tensor
   */
  void Decompress(ByteBuf compressed, int dtype,
                  ByteBuf& decompressed) override;

 private:
  size_t Packing(void* data, size_t len, int dtype);

  template <typename T>
  size_t _Packing(T* data, size_t len);

  size_t Unpacking(void* dst, const void* src, size_t len, int dtype);

  template <typename T1, typename T2>
  size_t _Unpacking(T1* dst, const T2* src, size_t size);

  size_t PackingCuda(void* data, size_t len, int dtype);
  size_t UnpackingCuda(void* dst, size_t src_len, const void* src, size_t len, int dtype);

  bool _use_scale;

#ifdef BYTEPS_ENABLE_CUDA
  float* _dev_out;
#endif
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_STRAT_ONEBIT_H