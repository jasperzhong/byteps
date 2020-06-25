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

#ifndef BYTEPS_COMPRESSOR_COMPRESSOR_H
#define BYTEPS_COMPRESSOR_COMPRESSOR_H

#include <memory>

#include "../cpu_reducer.h"
#include "common.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief Compressor interface
 * Compressor defines two universal API - \sa Compress & Decompress
 * 
 * \par
 * The caller do not need to allocate additional memory to store compressed data
 * because there is an internal buffer to store the compressed data and the
 * pointer will be returned to the caller. Then the caller can send the returned
 * compressed data.
 * 
 * \par
 * Detailed impl 
 *
 *
 * \sa tensor_t
 */
class Compressor {
 public:
  Compressor(size_t size)
      : _size(size),
        _buf(new byte_t[size]),
        _cpu_reducer(new CpuReducer(nullptr)){};
  virtual ~Compressor() = default;

  /*!
   * \brief Compress function
   *
   * \note except for error-feedback and momentum, the underlying data of input
   * should never be changed. this is because input is still used in error
   * feedback if enabled.
   *
   * \param grad gradient tensor, passed by value.
   * \param compressed Output compressed tensor, passed by ref. Passed
   * `compressed` can be an empty tensor, its values are given in this function.
   * In the implementation, its pointer must be assigned to the buffer of the
   * compressor (`_buf`).
   */
  virtual void Compress(tensor_t grad, tensor_t& compressed) {
    compressed.data = _buf.get();  // this is a must
  };

  /*!
   * \brief Decompress function
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  virtual void Decompress(tensor_t compressed, tensor_t& decompressed) {}

 protected:
  /*!
   * \brief faster version of `UpdateError` via operation fusion
   *
   * \par
   * This is a helper function implemented by each compressor. If defined,
   * `ErrorFeedback` will use this function instead of defualt `UpdateError`
   * function implemented in error_feedback.cc. If undefined, default
   * `UpdateError` will be used.
   *
   * \par
   * Typically `UpdateError` needs to decompress and do a substraction. But for
   * most compressors, the step of decompression can be avoided. For example,
   * for topk compressor, `UpdateError` can be simplied in this way:
   * 1. e <- p (e is the error and p is the corrected gradient)
   * 2. zero-fill e with selected k indices
   *
   * Actually it is a fusion of original decompression and substraction.
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  virtual void FastUpdateError(tensor_t error, tensor_t corrected,
                               tensor_t compressed) {
    BPS_LOG(FATAL) << "FastUpdateError is not implemented";
  };

  /*!
   * \brief buffer to store compressed grad
   */
  std::unique_ptr<byte_t[]> _buf;

  /*!
   * \brief original size
   */
  size_t _size;

  /*!
   * \brief CPU reducer
   */
  std::unique_ptr<CpuReducer> _cpu_reducer;
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_COMPRESSOR_H