#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "aligned_alloc.h"
#include "gguf_loader.h"

namespace cieft {

struct TensorF32 {
  std::vector<std::uint64_t> dims;
  std::uint64_t numel = 0;
  AlignedBuffer storage;

  float* data() { return static_cast<float*>(storage.data()); }
  const float* data() const { return static_cast<const float*>(storage.data()); }
};

struct GlobalWeights {
  TensorF32 token_embd;  // [d_model, vocab]
  std::optional<TensorF32> output_norm;  // [d_model]
  std::optional<TensorF32> output;       // [d_model, vocab]
};

struct LayerWeights {
  std::uint32_t index = 0;

  TensorF32 attn_norm;    // [d_model]
  TensorF32 attn_q;       // [d_model, d_model]
  TensorF32 attn_k;       // [d_model, kv_dim]
  TensorF32 attn_v;       // [d_model, kv_dim]
  TensorF32 attn_output;  // [d_model, d_model]

  TensorF32 ffn_norm;  // [d_model]
  TensorF32 ffn_gate;  // [d_model, ffn_hidden]
  TensorF32 ffn_up;    // [d_model, ffn_hidden]
  TensorF32 ffn_down;  // [ffn_hidden, d_model]
};

struct Weights {
  ModelConfig cfg;
  GlobalWeights global;
  std::vector<LayerWeights> layers;
};

TensorF32 load_tensor_as_f32(const GGUFLoader& loader, std::string_view name, std::size_t alignment = 64);

Weights load_weights(const GGUFLoader& loader,
                     const std::vector<std::uint32_t>& layer_indices,
                     bool load_lm_head,
                     std::size_t alignment = 64);

// `W` is stored as [dim, vocab] with contiguous columns.
void gather_column(const TensorF32& W_dim_vocab, std::uint32_t token_id, float* out_dim);

}  // namespace cieft

