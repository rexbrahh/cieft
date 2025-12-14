#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cieft::gguf {

enum class ValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

struct ArraySummary {
  ValueType elem_type{};
  std::uint64_t length = 0;
};

struct Value {
  ValueType type{};
  std::variant<std::uint8_t,
               std::int8_t,
               std::uint16_t,
               std::int16_t,
               std::uint32_t,
               std::int32_t,
               std::uint64_t,
               std::int64_t,
               float,
               double,
               bool,
               std::string,
               ArraySummary>
      payload;
};

struct KV {
  std::string key;
  Value value;
};

struct Header {
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_kv_count = 0;
};

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t ggml_type = 0;
  std::uint64_t offset = 0;  // relative to data section start
};

struct File {
  Header header;
  std::vector<KV> metadata;
  std::vector<TensorInfo> tensors;
  std::size_t data_section_offset = 0;  // absolute file offset

  std::unordered_map<std::string, std::size_t> tensor_index_by_name;
  std::unordered_map<std::string, std::size_t> kv_index_by_key;
};

struct GGMLTypeTraits {
  const char* name = nullptr;  // e.g. "F32", "Q4_K"
  std::uint32_t block_size = 0;
  std::uint32_t type_size = 0;  // bytes per block
};

std::optional<GGMLTypeTraits> ggml_type_traits(std::uint32_t ggml_type);
std::optional<std::uint64_t> tensor_nbytes(const TensorInfo& t);

std::string value_type_to_string(ValueType t);
std::string value_to_string(const Value& v, std::size_t max_string_len = 160);

File parse(const std::uint8_t* data, std::size_t size);

}  // namespace cieft::gguf
