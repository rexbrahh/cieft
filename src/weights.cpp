#include "weights.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml_fp16.h"
#include "ggml_quants.h"

namespace cieft {

namespace {

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > std::numeric_limits<std::uint64_t>::max() / b) {
    throw std::runtime_error("u64 overflow");
  }
  return a * b;
}

std::uint64_t numel_u64(const std::vector<std::uint64_t>& dims) {
  std::uint64_t n = 1;
  for (const auto d : dims) {
    n = checked_mul_u64(n, d);
  }
  return n;
}

void expect_dims(const TensorView& t, const std::vector<std::uint64_t>& expected) {
  if (t.dims != expected) {
    throw std::runtime_error("unexpected shape for tensor " + std::string(t.name));
  }
}

TensorF32 allocate_f32(const std::vector<std::uint64_t>& dims, std::size_t alignment) {
  const std::uint64_t n = numel_u64(dims);
  const std::uint64_t bytes_u64 = checked_mul_u64(n, sizeof(float));
  if (bytes_u64 > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("tensor too large for this process");
  }
  TensorF32 out;
  out.dims = dims;
  out.numel = n;
  out.storage = AlignedBuffer::allocate(static_cast<std::size_t>(bytes_u64), alignment);
  return out;
}

std::uint64_t product_tail_u64(const std::vector<std::uint64_t>& dims, std::size_t start) {
  std::uint64_t n = 1;
  for (std::size_t i = start; i < dims.size(); i++) {
    n = checked_mul_u64(n, dims[i]);
  }
  return n;
}

}  // namespace

TensorF32 load_tensor_as_f32(const GGUFLoader& loader, std::string_view name, std::size_t alignment) {
  const auto t = loader.get_tensor(name);

  if (t.dims.empty()) {
    throw std::runtime_error("tensor has no dims: " + std::string(name));
  }

  TensorF32 out = allocate_f32(t.dims, alignment);

  // F32
  if (t.ggml_type == 0) {
    const std::uint64_t expected_bytes = checked_mul_u64(out.numel, sizeof(float));
    if (t.nbytes < expected_bytes) {
      throw std::runtime_error("tensor truncated: " + std::string(name));
    }
    std::memcpy(out.data(), t.data, static_cast<std::size_t>(expected_bytes));
    return out;
  }

  // F16 -> F32
  if (t.ggml_type == 1) {
    const std::uint64_t expected_bytes = checked_mul_u64(out.numel, sizeof(std::uint16_t));
    if (t.nbytes < expected_bytes) {
      throw std::runtime_error("tensor truncated: " + std::string(name));
    }
    const auto* src = reinterpret_cast<const std::uint16_t*>(t.data);
    float* dst = out.data();
    for (std::uint64_t i = 0; i < out.numel; i++) {
      dst[i] = ggml::fp16_to_fp32(src[i]);
    }
    return out;
  }

  // Q4_K -> F32
  if (t.ggml_type == 12) {
    const std::uint64_t row_len = t.dims[0];
    if (row_len % ggml::QK_K != 0) {
      throw std::runtime_error("Q4_K row_len not multiple of 256: " + std::string(name));
    }
    const std::uint64_t n_rows = product_tail_u64(t.dims, 1);
    const std::uint64_t blocks_per_row = row_len / ggml::QK_K;
    const std::uint64_t row_bytes = checked_mul_u64(blocks_per_row, sizeof(ggml::block_q4_K));
    const std::uint64_t expected_bytes = checked_mul_u64(row_bytes, n_rows);
    if (t.nbytes < expected_bytes) {
      throw std::runtime_error("tensor truncated: " + std::string(name));
    }

    const std::uint8_t* src_bytes = t.data;
    float* dst = out.data();
    for (std::uint64_t r = 0; r < n_rows; r++) {
      const auto* row = reinterpret_cast<const ggml::block_q4_K*>(src_bytes + r * row_bytes);
      ggml::dequantize_row_q4_k(row, dst + r * row_len, static_cast<std::int64_t>(row_len));
    }
    return out;
  }

  // Q6_K -> F32
  if (t.ggml_type == 14) {
    const std::uint64_t row_len = t.dims[0];
    if (row_len % ggml::QK_K != 0) {
      throw std::runtime_error("Q6_K row_len not multiple of 256: " + std::string(name));
    }
    const std::uint64_t n_rows = product_tail_u64(t.dims, 1);
    const std::uint64_t blocks_per_row = row_len / ggml::QK_K;
    const std::uint64_t row_bytes = checked_mul_u64(blocks_per_row, sizeof(ggml::block_q6_K));
    const std::uint64_t expected_bytes = checked_mul_u64(row_bytes, n_rows);
    if (t.nbytes < expected_bytes) {
      throw std::runtime_error("tensor truncated: " + std::string(name));
    }

    const std::uint8_t* src_bytes = t.data;
    float* dst = out.data();
    for (std::uint64_t r = 0; r < n_rows; r++) {
      const auto* row = reinterpret_cast<const ggml::block_q6_K*>(src_bytes + r * row_bytes);
      ggml::dequantize_row_q6_k(row, dst + r * row_len, static_cast<std::int64_t>(row_len));
    }
    return out;
  }

  throw std::runtime_error("unsupported ggml_type " + std::to_string(t.ggml_type) + " for tensor " + std::string(name));
}

Weights load_weights(const GGUFLoader& loader,
                     const std::vector<std::uint32_t>& layer_indices,
                     bool load_lm_head,
                     std::size_t alignment) {
  Weights w;
  w.cfg = loader.config();
  if (w.cfg.n_layers == 0 || w.cfg.d_model == 0 || w.cfg.n_heads == 0) {
    throw std::runtime_error("model config missing required metadata");
  }
  if (w.cfg.head_dim == 0 || w.cfg.kv_dim == 0) {
    throw std::runtime_error("invalid head config");
  }
  if (w.cfg.ffn_hidden_dim == 0) {
    throw std::runtime_error("missing llama.feed_forward_length");
  }

  // Globals
  w.global.token_embd = load_tensor_as_f32(loader, "token_embd.weight", alignment);
  if (w.global.token_embd.dims.size() != 2) {
    throw std::runtime_error("token_embd.weight is not 2D");
  }
  if (w.cfg.vocab_size == 0) {
    if (w.global.token_embd.dims[1] > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("vocab too large");
    }
    w.cfg.vocab_size = static_cast<std::uint32_t>(w.global.token_embd.dims[1]);
  }
  expect_dims(loader.get_tensor("token_embd.weight"), {w.cfg.d_model, w.cfg.vocab_size});

  if (load_lm_head) {
    w.global.output_norm = load_tensor_as_f32(loader, "output_norm.weight", alignment);
    expect_dims(loader.get_tensor("output_norm.weight"), {w.cfg.d_model});

    w.global.output = load_tensor_as_f32(loader, "output.weight", alignment);
    expect_dims(loader.get_tensor("output.weight"), {w.cfg.d_model, w.cfg.vocab_size});
  }

  // Layers
  w.layers.reserve(layer_indices.size());
  for (const auto i : layer_indices) {
    if (i >= w.cfg.n_layers) {
      throw std::runtime_error("layer index out of range");
    }

    LayerWeights lw;
    lw.index = i;

    const std::string prefix = "blk." + std::to_string(i) + ".";
    lw.attn_norm = load_tensor_as_f32(loader, prefix + "attn_norm.weight", alignment);
    lw.attn_q = load_tensor_as_f32(loader, prefix + "attn_q.weight", alignment);
    lw.attn_k = load_tensor_as_f32(loader, prefix + "attn_k.weight", alignment);
    lw.attn_v = load_tensor_as_f32(loader, prefix + "attn_v.weight", alignment);
    lw.attn_output = load_tensor_as_f32(loader, prefix + "attn_output.weight", alignment);

    lw.ffn_norm = load_tensor_as_f32(loader, prefix + "ffn_norm.weight", alignment);
    lw.ffn_gate = load_tensor_as_f32(loader, prefix + "ffn_gate.weight", alignment);
    lw.ffn_up = load_tensor_as_f32(loader, prefix + "ffn_up.weight", alignment);
    lw.ffn_down = load_tensor_as_f32(loader, prefix + "ffn_down.weight", alignment);

    // Shape checks (match the spec you provided).
    expect_dims(loader.get_tensor(prefix + "attn_norm.weight"), {w.cfg.d_model});
    expect_dims(loader.get_tensor(prefix + "attn_q.weight"), {w.cfg.d_model, w.cfg.d_model});
    expect_dims(loader.get_tensor(prefix + "attn_k.weight"), {w.cfg.d_model, w.cfg.kv_dim});
    expect_dims(loader.get_tensor(prefix + "attn_v.weight"), {w.cfg.d_model, w.cfg.kv_dim});
    expect_dims(loader.get_tensor(prefix + "attn_output.weight"), {w.cfg.d_model, w.cfg.d_model});

    expect_dims(loader.get_tensor(prefix + "ffn_norm.weight"), {w.cfg.d_model});
    expect_dims(loader.get_tensor(prefix + "ffn_gate.weight"), {w.cfg.d_model, w.cfg.ffn_hidden_dim});
    expect_dims(loader.get_tensor(prefix + "ffn_up.weight"), {w.cfg.d_model, w.cfg.ffn_hidden_dim});
    expect_dims(loader.get_tensor(prefix + "ffn_down.weight"), {w.cfg.ffn_hidden_dim, w.cfg.d_model});

    w.layers.push_back(std::move(lw));
  }

  return w;
}

void gather_column(const TensorF32& W_dim_vocab, std::uint32_t token_id, float* out_dim) {
  if (W_dim_vocab.dims.size() != 2) {
    throw std::runtime_error("gather_column expects 2D tensor");
  }
  const std::uint64_t dim = W_dim_vocab.dims[0];
  const std::uint64_t vocab = W_dim_vocab.dims[1];
  if (token_id >= vocab) {
    throw std::runtime_error("token_id out of range");
  }
  const float* src = W_dim_vocab.data() + static_cast<std::uint64_t>(token_id) * dim;
  std::memcpy(out_dim, src, static_cast<std::size_t>(dim) * sizeof(float));
}

}  // namespace cieft
