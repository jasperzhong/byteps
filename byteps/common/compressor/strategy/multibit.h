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

#ifndef BYTEPS_COMPRESS_STRAT_MULTIBIT_H
#define BYTEPS_COMPRESS_STRAT_MULTIBIT_H

#include "../base_compressor.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Multibit Compressor
 * 
 * paper: QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
 * https://arxiv.org/pdf/1610.02132.pdf
 * 
 * quantize gradients into k levels
 * use stochastic rounding and elias encoding
 * 
 */
class MultibitCompressor : public BaseCompressor {
 public:
  explicit MultibitCompressor(int k);
  virtual ~MultibitCompressor();

  /*!
   * \brief Compress function
   * 
   * 1. normalize 
   * 2. stochastic rounding 
   * 3. elias encoding
   * 
   * \param grad gradient tensor
   * \param dtype data tyoe
   * \param compressed compressed tensor
   */
  void Compress(ByteBuf grad, int dtype, ByteBuf& compressed) override;
  
  /*!
   * \brief Decompress function
   * 
   * 1. decoding 
   * 2. scale
   * 
   * \param compressed compressed tensor
   * \param dtype data type
   * \param decompressed decompressed tensor
   */
  void Decompress(ByteBuf compressed, int dtype,
                  ByteBuf& decompressed) override;

 private:
  int _k;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESS_STRAT_MULTIBIT_H