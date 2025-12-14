#include "layer0.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "kernels/math.h"
#include "kernels/matvec.h"
#include "kernels/rmsnorm.h"
#include "kernels/softmax.h"

namespace cieft {

KVCacheLayer::KVCacheLayer(std::uint32_t n_kv_heads, std::uint32_t max_seq, std::uint32_t head_dim)
    : n_kv_heads_(n_kv_heads), max_seq_(max_seq), head_dim_(head_dim) {
  if (n_kv_heads_ == 0 || max_seq_ == 0 || head_dim_ == 0) {
    throw std::runtime_error("KVCacheLayer: invalid dimensions");
  }
  k_.assign(static_cast<std::size_t>(n_kv_heads_) * max_seq_ * head_dim_, 0.0f);
  v_.assign(static_cast<std::size_t>(n_kv_heads_) * max_seq_ * head_dim_, 0.0f);
}

float* KVCacheLayer::k_ptr(std::uint32_t kv_head, std::uint32_t pos) {
  if (kv_head >= n_kv_heads_ || pos >= max_seq_) {
    throw std::runtime_error("KVCacheLayer::k_ptr out of range");
  }
  return &k_[((static_cast<std::size_t>(kv_head) * max_seq_ + pos) * head_dim_)];
}

float* KVCacheLayer::v_ptr(std::uint32_t kv_head, std::uint32_t pos) {
  if (kv_head >= n_kv_heads_ || pos >= max_seq_) {
    throw std::runtime_error("KVCacheLayer::v_ptr out of range");
  }
  return &v_[((static_cast<std::size_t>(kv_head) * max_seq_ + pos) * head_dim_)];
}

const float* KVCacheLayer::k_ptr(std::uint32_t kv_head, std::uint32_t pos) const {
  if (kv_head >= n_kv_heads_ || pos >= max_seq_) {
    throw std::runtime_error("KVCacheLayer::k_ptr out of range");
  }
  return &k_[((static_cast<std::size_t>(kv_head) * max_seq_ + pos) * head_dim_)];
}

const float* KVCacheLayer::v_ptr(std::uint32_t kv_head, std::uint32_t pos) const {
  if (kv_head >= n_kv_heads_ || pos >= max_seq_) {
    throw std::runtime_error("KVCacheLayer::v_ptr out of range");
  }
  return &v_[((static_cast<std::size_t>(kv_head) * max_seq_ + pos) * head_dim_)];
}

void KVCacheLayer::write(std::uint32_t pos, const float* k_kv_dim, const float* v_kv_dim) {
  if (pos >= max_seq_) {
    throw std::runtime_error("KVCacheLayer::write pos out of range");
  }
  for (std::uint32_t h = 0; h < n_kv_heads_; h++) {
    std::memcpy(k_ptr(h, pos), k_kv_dim + static_cast<std::size_t>(h) * head_dim_, head_dim_ * sizeof(float));
    std::memcpy(v_ptr(h, pos), v_kv_dim + static_cast<std::size_t>(h) * head_dim_, head_dim_ * sizeof(float));
  }
}

Layer0Context::Layer0Context(const ModelConfig& cfg) : cfg_(cfg) {
  if (cfg_.d_model == 0 || cfg_.n_heads == 0 || cfg_.head_dim == 0 || cfg_.n_kv_heads == 0 || cfg_.kv_dim == 0 ||
      cfg_.ffn_hidden_dim == 0) {
    throw std::runtime_error("Layer0Context: invalid model config");
  }
  if (cfg_.n_heads % cfg_.n_kv_heads != 0) {
    throw std::runtime_error("Layer0Context: n_heads must be divisible by n_kv_heads");
  }

  rope_.reset(cfg_.rope_dim != 0 ? cfg_.rope_dim : cfg_.head_dim, cfg_.rope_theta != 0.0f ? cfg_.rope_theta : 10000.0f);

  const std::uint32_t max_seq = cfg_.context_length != 0 ? cfg_.context_length : 2048;
  cache_ = KVCacheLayer(cfg_.n_kv_heads, max_seq, cfg_.head_dim);

  x_norm_.resize(cfg_.d_model);
  q_.resize(cfg_.d_model);
  k_.resize(cfg_.kv_dim);
  v_.resize(cfg_.kv_dim);
  attn_out_.resize(cfg_.d_model);
  tmp_d_model_.resize(cfg_.d_model);
  gate_.resize(cfg_.ffn_hidden_dim);
  up_.resize(cfg_.ffn_hidden_dim);
  attn_probs_.resize(max_seq);
}

void Layer0Context::step(const LayerWeights& layer, std::uint32_t pos, float* x_d_model) {
  if (pos >= cache_.max_seq()) {
    throw std::runtime_error("Layer0Context::step pos out of range");
  }
  const std::size_t d_model = cfg_.d_model;

  // ---- Attention ----
  kernels::rmsnorm_f32(x_d_model, layer.attn_norm.data(), d_model, cfg_.rms_epsilon, x_norm_.data());

  kernels::matvec_colmajor_f32(layer.attn_q.data(), cfg_.d_model, cfg_.d_model, x_norm_.data(), q_.data());
  kernels::matvec_colmajor_f32(layer.attn_k.data(), cfg_.d_model, cfg_.kv_dim, x_norm_.data(), k_.data());
  kernels::matvec_colmajor_f32(layer.attn_v.data(), cfg_.d_model, cfg_.kv_dim, x_norm_.data(), v_.data());

  rope_.apply_inplace(q_.data(), cfg_.n_heads, cfg_.head_dim, pos);
  rope_.apply_inplace(k_.data(), cfg_.n_kv_heads, cfg_.head_dim, pos);

  cache_.write(pos, k_.data(), v_.data());

  const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
  kernels::set_zero(attn_out_.data(), d_model);

  const std::uint32_t group = cfg_.n_heads / cfg_.n_kv_heads;
  for (std::uint32_t h = 0; h < cfg_.n_heads; h++) {
    const std::uint32_t kv_head = h / group;
    const float* qh = q_.data() + static_cast<std::size_t>(h) * cfg_.head_dim;

    float* probs = attn_probs_.data();
    for (std::uint32_t t = 0; t <= pos; t++) {
      const float* kh = cache_.k_ptr(kv_head, t);
      probs[t] = kernels::dot_f32(qh, kh, cfg_.head_dim) * inv_sqrt_hd;
    }

    kernels::softmax_inplace_f32(probs, static_cast<std::size_t>(pos + 1));

    float* out_h = attn_out_.data() + static_cast<std::size_t>(h) * cfg_.head_dim;
    kernels::set_zero(out_h, cfg_.head_dim);
    for (std::uint32_t t = 0; t <= pos; t++) {
      const float p = probs[t];
      const float* vh = cache_.v_ptr(kv_head, t);
      for (std::uint32_t i = 0; i < cfg_.head_dim; i++) {
        out_h[i] += p * vh[i];
      }
    }
  }

  kernels::matvec_colmajor_f32(layer.attn_output.data(), cfg_.d_model, cfg_.d_model, attn_out_.data(), tmp_d_model_.data());
  kernels::add_inplace(x_d_model, tmp_d_model_.data(), d_model);

  // ---- FFN ----
  kernels::rmsnorm_f32(x_d_model, layer.ffn_norm.data(), d_model, cfg_.rms_epsilon, x_norm_.data());

  kernels::matvec_colmajor_f32(layer.ffn_gate.data(), cfg_.d_model, cfg_.ffn_hidden_dim, x_norm_.data(), gate_.data());
  kernels::matvec_colmajor_f32(layer.ffn_up.data(), cfg_.d_model, cfg_.ffn_hidden_dim, x_norm_.data(), up_.data());

  for (std::uint32_t i = 0; i < cfg_.ffn_hidden_dim; i++) {
    gate_[i] = kernels::silu(gate_[i]) * up_[i];
  }

  kernels::matvec_colmajor_f32(layer.ffn_down.data(), cfg_.ffn_hidden_dim, cfg_.d_model, gate_.data(), tmp_d_model_.data());
  kernels::add_inplace(x_d_model, tmp_d_model_.data(), d_model);
}

}  // namespace cieft

