#pragma once

#include <bit>
#include <cstdint>

namespace cieft::ggml {

inline float fp16_to_fp32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h & 0x7C00u) >> 10;
  std::uint32_t mant = h & 0x03FFu;

  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/-0
    } else {
      // Subnormal: normalize.
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x03FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    // Inf/NaN
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  return std::bit_cast<float>(bits);
}

}  // namespace cieft::ggml

