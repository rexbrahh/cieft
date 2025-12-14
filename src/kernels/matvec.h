#pragma once

#include <cstddef>
#include <cstdint>

namespace cieft::kernels {

// Matrix `W` is stored as [in_dim, out_dim] with contiguous columns (dim0 contiguous),
// i.e. column j starts at W + j*in_dim. Computes y[out] = W^T * x[in].
inline void matvec_colmajor_f32(const float* W_in_out,
                                std::uint32_t in_dim,
                                std::uint32_t out_dim,
                                const float* x_in,
                                float* y_out) {
  for (std::uint32_t j = 0; j < out_dim; j++) {
    const float* col = W_in_out + static_cast<std::size_t>(j) * in_dim;
    double sum = 0.0;
    for (std::uint32_t i = 0; i < in_dim; i++) {
      sum += static_cast<double>(x_in[i]) * static_cast<double>(col[i]);
    }
    y_out[j] = static_cast<float>(sum);
  }
}

}  // namespace cieft::kernels

