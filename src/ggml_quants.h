#pragma once

#include <cstddef>
#include <cstdint>

namespace cieft::ggml {

constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;

struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[K_SCALE_SIZE];
  std::uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144);

struct block_q6_K {
  std::uint8_t ql[QK_K / 2];
  std::uint8_t qh[QK_K / 4];
  std::int8_t scales[QK_K / 16];
  std::uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210);

void dequantize_row_q4_k(const block_q4_K* x, float* y, std::int64_t k);
void dequantize_row_q6_k(const block_q6_K* x, float* y, std::int64_t k);

}  // namespace cieft::ggml

