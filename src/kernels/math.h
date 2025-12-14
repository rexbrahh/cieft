#pragma once

#include <cmath>
#include <cstddef>

namespace cieft::kernels {

inline void add_inplace(float* a, const float* b, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    a[i] += b[i];
  }
}

inline void set_zero(float* a, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    a[i] = 0.0f;
  }
}

inline float dot_f32(const float* a, const float* b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; i++) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
}

inline float silu(float x) {
  // x / (1 + exp(-x))
  return x / (1.0f + std::exp(-x));
}

}  // namespace cieft::kernels

