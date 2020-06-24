// Copyright 2020 Amazon Inc. or its affiliates. All Rights Reserved.
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

#include "momentum.h"

namespace byteps {
namespace common {
namespace compressor {

void Momentum::Compress(tensor_t grad, tensor_t& compressed) {
  // m_t = \mu * m_{t-1} + g_t
  UpdateMom(grad);

  // p_t = \mu m_t + g_t
  UpdateGradient(grad);

  // compress
  _cptr->Compress(grad, compressed);
}

void Momentum::Decompress(tensor_t compressed, tensor_t& decompressed) {
  _cptr->Decompress(compressed, decompressed);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps