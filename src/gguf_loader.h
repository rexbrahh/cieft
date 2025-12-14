#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gguf.h"
#include "mapped_file.h"

namespace cieft {

struct ModelConfig {
  std::uint32_t n_layers = 0;
  std::uint32_t d_model = 0;
  std::uint32_t n_heads = 0;
  std::uint32_t n_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t kv_dim = 0;
  std::uint32_t ffn_hidden_dim = 0;
  std::uint32_t vocab_size = 0;
  std::uint32_t context_length = 0;
  std::uint32_t rope_dim = 0;
  float rope_theta = 0.0f;
  float rms_epsilon = 0.0f;
};

struct TensorView {
  std::string_view name;
  std::vector<std::uint64_t> dims;
  std::uint32_t ggml_type = 0;
  const std::uint8_t* data = nullptr;  // tensor bytes in file (not dequantized)
  std::uint64_t nbytes = 0;
  std::uint64_t file_offset = 0;  // absolute file offset
};

class GGUFLoader {
 public:
  explicit GGUFLoader(const std::string& path);

  const gguf::File& file() const { return gguf_; }
  const MappedFile& mapped_file() const { return mapped_; }

  std::optional<TensorView> maybe_get_tensor(std::string_view name) const;
  TensorView get_tensor(std::string_view name) const;

  std::optional<std::uint32_t> kv_u32(std::string_view key) const;
  std::optional<std::uint64_t> kv_u64(std::string_view key) const;
  std::optional<float> kv_f32(std::string_view key) const;
  std::optional<std::string_view> kv_string(std::string_view key) const;

  ModelConfig config() const;

 private:
  MappedFile mapped_;
  gguf::File gguf_;
  std::vector<std::uint64_t> tensor_size_from_offsets_;
};

}  // namespace cieft

