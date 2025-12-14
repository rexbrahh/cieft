#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cieft::kernels {

class RoPECache {
 public:
  RoPECache() = default;

  RoPECache(std::uint32_t rope_dim, float theta) { reset(rope_dim, theta); }

  void reset(std::uint32_t rope_dim, float theta) {
    if (rope_dim == 0 || (rope_dim % 2) != 0) {
      throw std::runtime_error("rope_dim must be non-zero and even");
    }
    if (!(theta > 0.0f)) {
      throw std::runtime_error("rope theta must be > 0");
    }
    rope_dim_ = rope_dim;
    theta_ = theta;
    inv_freq_.resize(rope_dim_ / 2);

    for (std::uint32_t i = 0; i < rope_dim_ / 2; i++) {
      const float exponent = static_cast<float>(2.0 * static_cast<double>(i) / static_cast<double>(rope_dim_));
      inv_freq_[i] = std::pow(theta_, -exponent);
    }
  }

  std::uint32_t rope_dim() const { return rope_dim_; }
  float theta() const { return theta_; }

  // Applies RoPE to the first `rope_dim` dims of each head vector.
  void apply_inplace(float* x, std::uint32_t n_heads, std::uint32_t head_dim, std::uint32_t pos) const {
    if (rope_dim_ == 0) {
      throw std::runtime_error("RoPECache not initialized");
    }
    if (rope_dim_ > head_dim) {
      throw std::runtime_error("rope_dim > head_dim");
    }
    if ((rope_dim_ % 2) != 0) {
      throw std::runtime_error("rope_dim must be even");
    }

    for (std::uint32_t h = 0; h < n_heads; h++) {
      float* head = x + static_cast<std::size_t>(h) * head_dim;
      for (std::uint32_t i = 0; i < rope_dim_ / 2; i++) {
        const float angle = static_cast<float>(pos) * inv_freq_[i];
        const float c = std::cos(angle);
        const float s = std::sin(angle);

        const std::size_t idx0 = static_cast<std::size_t>(2) * i;
        const std::size_t idx1 = idx0 + 1;

        const float v0 = head[idx0];
        const float v1 = head[idx1];
        head[idx0] = v0 * c - v1 * s;
        head[idx1] = v0 * s + v1 * c;
      }
    }
  }

 private:
  std::uint32_t rope_dim_ = 0;
  float theta_ = 0.0f;
  std::vector<float> inv_freq_;
};

}  // namespace cieft::kernels

