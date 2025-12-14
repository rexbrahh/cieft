#pragma once

#include <cstdint>
#include <vector>

#include "gguf_loader.h"
#include "kernels/rope.h"
#include "weights.h"

namespace cieft {

class KVCacheLayer {
 public:
  KVCacheLayer() = default;
  KVCacheLayer(std::uint32_t n_kv_heads, std::uint32_t max_seq, std::uint32_t head_dim);

  std::uint32_t n_kv_heads() const { return n_kv_heads_; }
  std::uint32_t max_seq() const { return max_seq_; }
  std::uint32_t head_dim() const { return head_dim_; }

  float* k_ptr(std::uint32_t kv_head, std::uint32_t pos);
  float* v_ptr(std::uint32_t kv_head, std::uint32_t pos);
  const float* k_ptr(std::uint32_t kv_head, std::uint32_t pos) const;
  const float* v_ptr(std::uint32_t kv_head, std::uint32_t pos) const;

  void write(std::uint32_t pos, const float* k_kv_dim, const float* v_kv_dim);

 private:
  std::uint32_t n_kv_heads_ = 0;
  std::uint32_t max_seq_ = 0;
  std::uint32_t head_dim_ = 0;
  std::vector<float> k_;
  std::vector<float> v_;
};

class Layer0Context {
 public:
  explicit Layer0Context(const ModelConfig& cfg);

  // Updates K/V cache at `pos` and runs one layer forward in-place on `x` (length d_model).
  void step(const LayerWeights& layer, std::uint32_t pos, float* x_d_model);

 private:
  ModelConfig cfg_;
  kernels::RoPECache rope_;
  KVCacheLayer cache_;

  std::vector<float> x_norm_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> attn_out_;
  std::vector<float> tmp_d_model_;
  std::vector<float> gate_;
  std::vector<float> up_;
  std::vector<float> attn_probs_;
};

}  // namespace cieft

