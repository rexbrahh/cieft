#pragma once

#include <cmath>
#include <cstddef>

namespace cieft::kernels {

inline void rmsnorm_f32(const float* x, const float* weight, std::size_t n, float eps, float* out) {
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n; i++) {
    const double v = x[i];
    sum_sq += v * v;
  }
  const double mean_sq = sum_sq / static_cast<double>(n);
  const float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean_sq) + eps);

  for (std::size_t i = 0; i < n; i++) {
    out[i] = x[i] * inv_rms * weight[i];
  }
}

}  // namespace cieft::kernels

