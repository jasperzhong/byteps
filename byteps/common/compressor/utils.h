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

#ifndef BYTEPS_COMPRESSOR_UTILS_H
#define BYTEPS_COMPRESSOR_UTILS_H

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "common.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief serialize key-vals hyper-params for network transmission
 *
 * \param kwargs hyper-params
 * \return std::string serialized data
 */
inline std::string Serialize(const kwargs_t& kwargs) {
  std::ostringstream os;
  os << kwargs.size();
  for (auto const& kwarg : kwargs) {
    os << " " << kwarg.first << " " << kwarg.second;
  }
  return os.str();
}

/*!
 * \brief deserialize serialized data into key-vals hyper-params
 *
 * \param content serialized data
 * \return kwargs_t hyper-params
 */
inline kwargs_t Deserialize(const std::string& content) {
  kwargs_t kwargs;
  std::istringstream is(content);
  size_t size = 0;
  is >> size;
  for (size_t i = 0; i < size; ++i) {
    kwargs_t::key_type key;
    kwargs_t::mapped_type val;
    is >> key >> val;
    kwargs[key] = val;
  }

  return kwargs;
}

/*!
 * \brief random number generator based on xorshift128plus
 *
 * refer to https://en.wikipedia.org/wiki/Xorshift#xorshift+
 */
class XorShift128PlusBitShifterRNG {
 public:
  XorShift128PlusBitShifterRNG() {
    std::random_device rd;
    _state = {rd(), rd()};
  }

  // uniform int among [low, high)
  uint64_t Randint(uint64_t low, uint64_t high) {
    return xorshift128p() % (high - low) + low;
  };

  // uniform [0, 1]
  double Rand() { return double(xorshift128p()) / MAX; }

  // Bernoulli Distributation
  bool Bernoulli(double p) { return xorshift128p() < uint64_t(p * MAX); }

  void set_seed(uint64_t seed) { _state = {seed, seed}; }

 private:
  struct xorshift128p_state {
    uint64_t a, b;
  };

  uint64_t xorshift128p() {
    uint64_t t = _state.a;
    uint64_t const s = _state.b;
    _state.a = s;
    t ^= t << 23;        // a
    t ^= t >> 17;        // b
    t ^= s ^ (s >> 26);  // c
    _state.b = t;
    return t + s;
  };

  xorshift128p_state _state;

  static constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
};

/*!
 * \brief Bit Writer
 *
 */
template <typename T>
class BitWriter {
 public:
  explicit BitWriter(T* data) : _dptr(data), _bits(0) {}
  void Put(bool x) {
    size_t pos = _bits / PACKING_SIZE;
    _dptr[pos] <<= 1;
    _dptr[pos] |= x;
    ++_bits;
  }

  void Pad() {
    size_t pos = _bits / PACKING_SIZE;
    size_t padding_size = (PACKING_SIZE - _bits % PACKING_SIZE) % PACKING_SIZE;
    _bits += padding_size;
    _dptr[pos] <<= padding_size;
  }

  size_t bits() const { return _bits; }
  size_t ints() const { return _bits / PACKING_SIZE; }

 private:
  static constexpr size_t PACKING_SIZE = sizeof(T) * 8;
  T* _dptr;  // allocated
  size_t _bits;
};

/*!
 * \brief Bit Reader
 *
 */
template <typename T>
class BitReader {
 public:
  explicit BitReader(const T* data) : _dptr(data), _bits(0) {}
  bool Get() {
    size_t pos = _bits / PACKING_SIZE;
    size_t offset = PACKING_SIZE - 1 - _bits % PACKING_SIZE;
    _bits++;
    return _dptr[pos] & (1 << offset);
  }

  size_t bits() const { return _bits; }

 private:
  static constexpr size_t PACKING_SIZE = sizeof(T) * 8;
  const T* _dptr;  // allocated
  size_t _bits;
};

inline uint32_t RoundNextPow2(uint32_t v) {
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v += 1;
  return v;
}

template <typename T>
void EliasDeltaEncode(BitWriter<T>& bit_writer, unsigned int x) {
  int len = 1 + std::floor(std::log2(x));
  int lenth_of_len = std::floor(std::log2(len));

  for (int i = lenth_of_len; i > 0; --i) bit_writer.Put(0);
  for (int i = lenth_of_len; i >= 0; --i) bit_writer.Put((len >> i) & 1);
  for (int i = len - 2; i >= 0; i--) bit_writer.Put((x >> i) & 1);
}

template <typename T>
unsigned int EliasDeltaDecode(BitReader<T>& bit_reader) {
  int num = 1;
  int len = 1;
  int lenth_of_len = 0;
  while (!bit_reader.Get()) lenth_of_len++;
  for (int i = 0; i < lenth_of_len; i++) {
    len <<= 1;
    if (bit_reader.Get()) len |= 1;
  }
  for (int i = 0; i < len - 1; i++) {
    num <<= 1;
    if (bit_reader.Get()) num |= 1;
  }
  return num;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_UTILS_H