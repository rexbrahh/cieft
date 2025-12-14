#include "ggml_quants.h"

#include <cassert>
#include <cstdint>

#include "ggml_fp16.h"

namespace cieft::ggml {

namespace {

inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

}  // namespace

void dequantize_row_q4_k(const block_q4_K* x, float* y, std::int64_t k) {
  assert(k % QK_K == 0);
  const int nb = static_cast<int>(k / QK_K);

  for (int i = 0; i < nb; i++) {
    const std::uint8_t* q = x[i].qs;

    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);

    int is = 0;
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;

      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * (q[l] & 0xF) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * (q[l] >> 4) - m2;
      }
      q += 32;
      is += 2;
    }
  }
}

}  // namespace cieft::ggml

