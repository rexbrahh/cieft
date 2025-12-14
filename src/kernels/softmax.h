#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace cieft::kernels {

inline void softmax_inplace_f32(float* x, std::size_t n) {
  if (n == 0) {
    return;
  }
  float max_v = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; i++) {
    if (x[i] > max_v) {
      max_v = x[i];
    }
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; i++) {
    const float e = std::exp(x[i] - max_v);
    x[i] = e;
    sum += e;
  }

  const float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
  for (std::size_t i = 0; i < n; i++) {
    x[i] *= inv_sum;
  }
}

}  // namespace cieft::kernels

