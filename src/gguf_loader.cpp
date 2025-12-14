#include "gguf_loader.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cieft {

namespace {

std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    throw std::runtime_error("u64 overflow");
  }
  return a + b;
}

std::optional<std::uint32_t> to_u32(const gguf::Value& v) {
  if (std::holds_alternative<std::uint32_t>(v.payload)) {
    return std::get<std::uint32_t>(v.payload);
  }
  if (std::holds_alternative<std::int32_t>(v.payload)) {
    const auto x = std::get<std::int32_t>(v.payload);
    if (x < 0) {
      return std::nullopt;
    }
    return static_cast<std::uint32_t>(x);
  }
  if (std::holds_alternative<std::uint64_t>(v.payload)) {
    const auto x = std::get<std::uint64_t>(v.payload);
    if (x > std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<std::uint32_t>(x);
  }
  if (std::holds_alternative<std::int64_t>(v.payload)) {
    const auto x = std::get<std::int64_t>(v.payload);
    if (x < 0 || x > std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<std::uint32_t>(x);
  }
  return std::nullopt;
}

std::optional<std::uint64_t> to_u64(const gguf::Value& v) {
  if (std::holds_alternative<std::uint64_t>(v.payload)) {
    return std::get<std::uint64_t>(v.payload);
  }
  if (std::holds_alternative<std::uint32_t>(v.payload)) {
    return static_cast<std::uint64_t>(std::get<std::uint32_t>(v.payload));
  }
  if (std::holds_alternative<std::int64_t>(v.payload)) {
    const auto x = std::get<std::int64_t>(v.payload);
    if (x < 0) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(x);
  }
  if (std::holds_alternative<std::int32_t>(v.payload)) {
    const auto x = std::get<std::int32_t>(v.payload);
    if (x < 0) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(x);
  }
  return std::nullopt;
}

std::optional<float> to_f32(const gguf::Value& v) {
  if (std::holds_alternative<float>(v.payload)) {
    return std::get<float>(v.payload);
  }
  if (std::holds_alternative<double>(v.payload)) {
    return static_cast<float>(std::get<double>(v.payload));
  }
  if (const auto u = to_u32(v)) {
    return static_cast<float>(*u);
  }
  if (const auto u = to_u64(v)) {
    return static_cast<float>(*u);
  }
  return std::nullopt;
}

}  // namespace

GGUFLoader::GGUFLoader(const std::string& path) : mapped_(path), gguf_(gguf::parse(mapped_.data(), mapped_.size())) {
  tensor_size_from_offsets_.assign(gguf_.tensors.size(), 0);

  std::vector<std::size_t> idx(gguf_.tensors.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    return gguf_.tensors[a].offset < gguf_.tensors[b].offset;
  });

  for (std::size_t i = 0; i < idx.size(); i++) {
    const std::size_t cur = idx[i];
    const std::uint64_t cur_abs = checked_add_u64(static_cast<std::uint64_t>(gguf_.data_section_offset),
                                                  gguf_.tensors[cur].offset);
    std::uint64_t next_abs = static_cast<std::uint64_t>(mapped_.size());
    if (i + 1 < idx.size()) {
      const std::size_t nxt = idx[i + 1];
      next_abs = checked_add_u64(static_cast<std::uint64_t>(gguf_.data_section_offset), gguf_.tensors[nxt].offset);
    }
    if (next_abs < cur_abs) {
      throw std::runtime_error("tensor offsets not monotonic");
    }
    tensor_size_from_offsets_[cur] = next_abs - cur_abs;
  }
}

std::optional<TensorView> GGUFLoader::maybe_get_tensor(std::string_view name) const {
  auto it = gguf_.tensor_index_by_name.find(std::string(name));
  if (it == gguf_.tensor_index_by_name.end()) {
    return std::nullopt;
  }
  const auto& ti = gguf_.tensors[it->second];

  const std::uint64_t abs_off = checked_add_u64(static_cast<std::uint64_t>(gguf_.data_section_offset), ti.offset);
  const auto nbytes = gguf::tensor_nbytes(ti).value_or(tensor_size_from_offsets_[it->second]);

  if (abs_off > mapped_.size() || abs_off + nbytes > mapped_.size()) {
    throw std::runtime_error("tensor view out of bounds: " + std::string(name));
  }

  return TensorView{
      .name = ti.name,
      .dims = ti.dims,
      .ggml_type = ti.ggml_type,
      .data = mapped_.data() + abs_off,
      .nbytes = nbytes,
      .file_offset = abs_off,
  };
}

TensorView GGUFLoader::get_tensor(std::string_view name) const {
  if (auto tv = maybe_get_tensor(name)) {
    return *tv;
  }
  throw std::runtime_error("tensor not found: " + std::string(name));
}

std::optional<std::uint32_t> GGUFLoader::kv_u32(std::string_view key) const {
  auto it = gguf_.kv_index_by_key.find(std::string(key));
  if (it == gguf_.kv_index_by_key.end()) {
    return std::nullopt;
  }
  return to_u32(gguf_.metadata[it->second].value);
}

std::optional<std::uint64_t> GGUFLoader::kv_u64(std::string_view key) const {
  auto it = gguf_.kv_index_by_key.find(std::string(key));
  if (it == gguf_.kv_index_by_key.end()) {
    return std::nullopt;
  }
  return to_u64(gguf_.metadata[it->second].value);
}

std::optional<float> GGUFLoader::kv_f32(std::string_view key) const {
  auto it = gguf_.kv_index_by_key.find(std::string(key));
  if (it == gguf_.kv_index_by_key.end()) {
    return std::nullopt;
  }
  return to_f32(gguf_.metadata[it->second].value);
}

std::optional<std::string_view> GGUFLoader::kv_string(std::string_view key) const {
  auto it = gguf_.kv_index_by_key.find(std::string(key));
  if (it == gguf_.kv_index_by_key.end()) {
    return std::nullopt;
  }
  const auto& v = gguf_.metadata[it->second].value;
  if (!std::holds_alternative<std::string>(v.payload)) {
    return std::nullopt;
  }
  return std::get<std::string>(v.payload);
}

ModelConfig GGUFLoader::config() const {
  ModelConfig cfg;
  cfg.n_layers = kv_u32("llama.block_count").value_or(0);
  cfg.d_model = kv_u32("llama.embedding_length").value_or(0);
  cfg.n_heads = kv_u32("llama.attention.head_count").value_or(0);
  cfg.n_kv_heads = kv_u32("llama.attention.head_count_kv").value_or(0);
  cfg.ffn_hidden_dim = kv_u32("llama.feed_forward_length").value_or(0);
  cfg.context_length = kv_u32("llama.context_length").value_or(0);
  cfg.rope_dim = kv_u32("llama.rope.dimension_count").value_or(0);
  cfg.rope_theta = kv_f32("llama.rope.freq_base").value_or(0.0f);
  cfg.rms_epsilon = kv_f32("llama.attention.layer_norm_rms_epsilon").value_or(0.0f);

  if (cfg.n_heads != 0 && cfg.d_model % cfg.n_heads == 0) {
    cfg.head_dim = cfg.d_model / cfg.n_heads;
  }
  cfg.kv_dim = cfg.n_kv_heads * cfg.head_dim;

  // Derive vocab size from embedding tensor shape if present.
  if (auto t = maybe_get_tensor("token_embd.weight")) {
    if (t->dims.size() == 2 && t->dims[0] <= std::numeric_limits<std::uint32_t>::max() &&
        t->dims[1] <= std::numeric_limits<std::uint32_t>::max()) {
      cfg.vocab_size = static_cast<std::uint32_t>(t->dims[1]);
    }
  }

  return cfg;
}

}  // namespace cieft
