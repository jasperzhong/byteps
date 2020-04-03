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

#include "base_compressor.h"

#include "../logging.h"

namespace byteps {
namespace common {
namespace compressor {

CompressorRegistry::map_t CompressorRegistry::_ctor_map;

CompressorRegistry::Register::Register(std::string name, ctor_t ctor) {
  BPS_CHECK_EQ(_ctor_map.count(name), 0)
      << "Duplicate registration of compressor under name " << name;
  _ctor_map.emplace(name + "_type", std::move(ctor));
}

CompressorRegistry::ctor_t CompressorRegistry::Find(const std::string& name) {
  auto it = _ctor_map.find(name);
  if (it == _ctor_map.end()) {
    BPS_LOG(WARNING) << "No compressor registered under name:" << name;
    return nullptr;
  }
  return it->second;
}

std::unique_ptr<BaseCompressor> CompressorRegistry::Create(
    const kwargs_t& kwargs) {
  const std::string types[] = {"error_feedback_type", "compressor_type"};

  for (auto& type : types) {
    auto iter = kwargs.find(type);
    if (iter != kwargs.end()) {
      auto ctor = CompressorRegistry::Find(iter->second + "_" + type);
      BPS_CHECK_NE(ctor, nullptr);
      return ctor(kwargs);
    }
  }

  return nullptr;
}

BaseCompressor::BaseCompressor() = default;

BaseCompressor::~BaseCompressor() {
#ifdef BYTEPS_ENABLE_CUDA
  cudaFree(_dev_buf);
  if (_stream) {
    cudaStreamDestroy(*_stream);
    delete _stream;
  }
#endif
};

void BaseCompressor::Init(size_t aligned_size) {
  _buf.reset(new char[aligned_size]);
  _cpu_reducer.reset(new CpuReducer(nullptr, aligned_size));
#ifdef BYTEPS_ENABLE_CUDA
  cudaMalloc(&_dev_buf, aligned_size);
  _stream = new cudaStream_t;
  cudaStreamCreate(_stream);
  _cpu_reducer->set_cuda_stream(_stream);
  BPS_LOG(INFO) << "cuda malloc for compressor  size" << aligned_size;;
#endif
}
}  // namespace compressor
}  // namespace common
}  // namespace byteps