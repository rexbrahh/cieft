#include "ggml_quants.h"

#include <cassert>
#include <cstdint>

#include "ggml_fp16.h"

namespace cieft::ggml {

void dequantize_row_q6_k(const block_q6_K* x, float* y, std::int64_t k) {
  assert(k % QK_K == 0);
  const std::int64_t nb = k / QK_K;

  for (std::int64_t i = 0; i < nb; i++) {
    const float d = fp16_to_fp32(x[i].d);

    const std::uint8_t* ql = x[i].ql;
    const std::uint8_t* qh = x[i].qh;
    const std::int8_t* sc = x[i].scales;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const std::int8_t q1 = static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const std::int8_t q2 = static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const std::int8_t q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const std::int8_t q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

}  // namespace cieft::ggml

