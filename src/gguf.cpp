#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "reader.h"

namespace cieft::gguf {

namespace {

constexpr std::uint32_t kDefaultAlignment = 32;

template <typename T>
bool mul_overflow_u64(std::uint64_t a, std::uint64_t b, T* out) {
  static_assert(std::is_unsigned_v<T>);
  if (a == 0 || b == 0) {
    *out = 0;
    return false;
  }
  if (a > std::numeric_limits<T>::max() / b) {
    return true;
  }
  *out = static_cast<T>(a * b);
  return false;
}

bool add_overflow_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    return true;
  }
  *out = a + b;
  return false;
}

void skip_u64(Reader& r, std::uint64_t nbytes) {
  if (nbytes > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("skip too large");
  }
  r.skip(static_cast<std::size_t>(nbytes));
}

Value read_value(Reader& r, ValueType t) {
  Value v;
  v.type = t;
  switch (t) {
    case ValueType::Uint8:
      v.payload = r.read<std::uint8_t>();
      return v;
    case ValueType::Int8:
      v.payload = r.read<std::int8_t>();
      return v;
    case ValueType::Uint16:
      v.payload = r.read<std::uint16_t>();
      return v;
    case ValueType::Int16:
      v.payload = r.read<std::int16_t>();
      return v;
    case ValueType::Uint32:
      v.payload = r.read<std::uint32_t>();
      return v;
    case ValueType::Int32:
      v.payload = r.read<std::int32_t>();
      return v;
    case ValueType::Uint64:
      v.payload = r.read<std::uint64_t>();
      return v;
    case ValueType::Int64:
      v.payload = r.read<std::int64_t>();
      return v;
    case ValueType::Float32:
      v.payload = r.read<float>();
      return v;
    case ValueType::Float64:
      v.payload = r.read<double>();
      return v;
    case ValueType::Bool:
      v.payload = static_cast<bool>(r.read<std::uint8_t>() != 0);
      return v;
    case ValueType::String:
      v.payload = r.read_string();
      return v;
    case ValueType::Array: {
      const auto elem_type = static_cast<ValueType>(r.read<std::uint32_t>());
      const std::uint64_t n = r.read<std::uint64_t>();
      v.payload = ArraySummary{.elem_type = elem_type, .length = n};

      // We only store a summary. Still must advance the cursor safely.
      switch (elem_type) {
        case ValueType::String:
          for (std::uint64_t i = 0; i < n; i++) {
            (void)r.read_string();
          }
          return v;
        case ValueType::Uint8:
        case ValueType::Int8:
        case ValueType::Bool:
          skip_u64(r, n);
          return v;
        case ValueType::Uint16:
        case ValueType::Int16:
          if (std::uint64_t bytes = 0; mul_overflow_u64<std::uint64_t>(n, 2, &bytes)) {
            throw std::runtime_error("array skip overflow");
          } else {
            skip_u64(r, bytes);
          }
          return v;
        case ValueType::Uint32:
        case ValueType::Int32:
        case ValueType::Float32:
          if (std::uint64_t bytes = 0; mul_overflow_u64<std::uint64_t>(n, 4, &bytes)) {
            throw std::runtime_error("array skip overflow");
          } else {
            skip_u64(r, bytes);
          }
          return v;
        case ValueType::Uint64:
        case ValueType::Int64:
        case ValueType::Float64:
          if (std::uint64_t bytes = 0; mul_overflow_u64<std::uint64_t>(n, 8, &bytes)) {
            throw std::runtime_error("array skip overflow");
          } else {
            skip_u64(r, bytes);
          }
          return v;
        case ValueType::Array:
          throw std::runtime_error("array-of-array not supported in gguf");
      }
      throw std::runtime_error("unknown gguf array element type");
    }
  }
  throw std::runtime_error("unknown gguf value type");
}

}  // namespace

std::string value_type_to_string(ValueType t) {
  switch (t) {
    case ValueType::Uint8:
      return "u8";
    case ValueType::Int8:
      return "i8";
    case ValueType::Uint16:
      return "u16";
    case ValueType::Int16:
      return "i16";
    case ValueType::Uint32:
      return "u32";
    case ValueType::Int32:
      return "i32";
    case ValueType::Uint64:
      return "u64";
    case ValueType::Int64:
      return "i64";
    case ValueType::Float32:
      return "f32";
    case ValueType::Float64:
      return "f64";
    case ValueType::Bool:
      return "bool";
    case ValueType::String:
      return "string";
    case ValueType::Array:
      return "array";
  }
  return "unknown";
}

std::string value_to_string(const Value& v, std::size_t max_string_len) {
  std::ostringstream oss;
  if (std::holds_alternative<std::string>(v.payload)) {
    const auto& s = std::get<std::string>(v.payload);
    if (s.size() <= max_string_len) {
      oss << s;
    } else {
      oss << s.substr(0, max_string_len) << "…";
    }
    return oss.str();
  }
  if (std::holds_alternative<ArraySummary>(v.payload)) {
    const auto& a = std::get<ArraySummary>(v.payload);
    oss << "array<" << value_type_to_string(a.elem_type) << ">[" << a.length << "]";
    return oss.str();
  }
  if (std::holds_alternative<float>(v.payload)) {
    oss << std::setprecision(9) << std::get<float>(v.payload);
    return oss.str();
  }
  if (std::holds_alternative<double>(v.payload)) {
    oss << std::setprecision(17) << std::get<double>(v.payload);
    return oss.str();
  }
  if (std::holds_alternative<bool>(v.payload)) {
    oss << (std::get<bool>(v.payload) ? "true" : "false");
    return oss.str();
  }

  std::visit(
      [&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
          oss << +x;
        }
      },
      v.payload);
  return oss.str();
}

std::optional<GGMLTypeTraits> ggml_type_traits(std::uint32_t ggml_type) {
  // Only implement what we actually need right now (Q4_K_M files + common floats).
  // Add more as they show up in your inspector output.
  switch (ggml_type) {
    case 0:  // GGML_TYPE_F32
      return GGMLTypeTraits{.name = "F32", .block_size = 1, .type_size = 4};
    case 1:  // GGML_TYPE_F16
      return GGMLTypeTraits{.name = "F16", .block_size = 1, .type_size = 2};
    case 12:  // GGML_TYPE_Q4_K
      // QK_K=256, sizeof(block_q4_K)=2*sizeof(ggml_half)+K_SCALE_SIZE+QK_K/2 = 144 bytes
      return GGMLTypeTraits{.name = "Q4_K", .block_size = 256, .type_size = 144};
    case 14:  // GGML_TYPE_Q6_K
      // QK_K=256, sizeof(block_q6_K)=sizeof(ggml_half)+QK_K/16+3*QK_K/4 = 210 bytes
      return GGMLTypeTraits{.name = "Q6_K", .block_size = 256, .type_size = 210};
    default:
      return std::nullopt;
  }
}

std::optional<std::uint64_t> tensor_nbytes(const TensorInfo& t) {
  const auto traits = ggml_type_traits(t.ggml_type);
  if (!traits) {
    return std::nullopt;
  }
  if (t.dims.empty()) {
    return 0;
  }

  // Elements are quantized in blocks along dim0.
  std::uint64_t blocks_dim0 = 0;
  if (traits->block_size == 1) {
    blocks_dim0 = t.dims[0];
  } else {
    const std::uint64_t bs = traits->block_size;
    blocks_dim0 = t.dims[0] / bs;
    if (t.dims[0] % bs != 0) {
      if (blocks_dim0 == std::numeric_limits<std::uint64_t>::max()) {
        return std::nullopt;
      }
      blocks_dim0 += 1;
    }
  }

  std::uint64_t nblocks = blocks_dim0;
  for (std::size_t i = 1; i < t.dims.size(); i++) {
    if (mul_overflow_u64<std::uint64_t>(nblocks, t.dims[i], &nblocks)) {
      return std::nullopt;
    }
  }

  std::uint64_t bytes = 0;
  if (mul_overflow_u64<std::uint64_t>(nblocks, traits->type_size, &bytes)) {
    return std::nullopt;
  }
  return bytes;
}

File parse(const std::uint8_t* data, std::size_t size) {
  Reader r(data, size);

  char magic[4]{};
  r.read_bytes(magic, sizeof(magic));
  if (!(magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F')) {
    throw std::runtime_error("not a GGUF file (bad magic)");
  }

  File out;
  out.header.version = r.read<std::uint32_t>();
  out.header.tensor_count = r.read<std::uint64_t>();
  out.header.metadata_kv_count = r.read<std::uint64_t>();

  out.metadata.reserve(static_cast<std::size_t>(out.header.metadata_kv_count));
  for (std::uint64_t i = 0; i < out.header.metadata_kv_count; i++) {
    KV kv;
    kv.key = r.read_string();
    const auto t = static_cast<ValueType>(r.read<std::uint32_t>());
    kv.value = read_value(r, t);

    out.kv_index_by_key.emplace(kv.key, out.metadata.size());
    out.metadata.push_back(std::move(kv));
  }

  out.tensors.reserve(static_cast<std::size_t>(out.header.tensor_count));
  for (std::uint64_t i = 0; i < out.header.tensor_count; i++) {
    TensorInfo ti;
    ti.name = r.read_string();
    const std::uint32_t n_dims = r.read<std::uint32_t>();
    ti.dims.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; d++) {
      ti.dims[d] = r.read<std::uint64_t>();
    }
    ti.ggml_type = r.read<std::uint32_t>();
    ti.offset = r.read<std::uint64_t>();

    out.tensor_index_by_name.emplace(ti.name, out.tensors.size());
    out.tensors.push_back(std::move(ti));
  }

  std::uint32_t alignment = kDefaultAlignment;
  if (auto it = out.kv_index_by_key.find("general.alignment"); it != out.kv_index_by_key.end()) {
    const auto& v = out.metadata[it->second].value;
    if (std::holds_alternative<std::uint32_t>(v.payload)) {
      alignment = std::get<std::uint32_t>(v.payload);
    } else if (std::holds_alternative<std::uint64_t>(v.payload)) {
      const auto a = std::get<std::uint64_t>(v.payload);
      if (a <= std::numeric_limits<std::uint32_t>::max()) {
        alignment = static_cast<std::uint32_t>(a);
      }
    }
  }
  out.data_section_offset = align_up(r.pos(), alignment);

  // Sanity check that tensor ranges fit in the file.
  if (out.data_section_offset > size) {
    throw std::runtime_error("data section offset out of bounds");
  }
  for (const auto& t : out.tensors) {
    std::uint64_t abs_off = 0;
    if (add_overflow_u64(static_cast<std::uint64_t>(out.data_section_offset), t.offset, &abs_off) ||
        abs_off > size) {
      throw std::runtime_error("tensor offset out of bounds: " + t.name);
    }

    if (const auto nbytes = tensor_nbytes(t)) {
      std::uint64_t end = 0;
      if (add_overflow_u64(abs_off, *nbytes, &end) || end > size) {
        throw std::runtime_error("tensor out of bounds: " + t.name);
      }
    }
  }

  return out;
}

}  // namespace cieft::gguf
