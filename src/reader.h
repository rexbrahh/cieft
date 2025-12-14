#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cieft {

class Reader {
 public:
  Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  std::size_t pos() const { return pos_; }
  std::size_t size() const { return size_; }

  void seek(std::size_t new_pos) {
    if (new_pos > size_) {
      throw std::runtime_error("seek past EOF");
    }
    pos_ = new_pos;
  }

  template <typename T>
  T read() {
    static_assert(std::is_trivially_copyable_v<T>);
    if (pos_ + sizeof(T) > size_) {
      throw std::runtime_error("read past EOF");
    }
    T out{};
    std::memcpy(&out, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return out;
  }

  void read_bytes(void* dst, std::size_t n) {
    if (pos_ + n > size_) {
      throw std::runtime_error("read_bytes past EOF");
    }
    std::memcpy(dst, data_ + pos_, n);
    pos_ += n;
  }

  void skip(std::size_t n) {
    if (pos_ + n > size_) {
      throw std::runtime_error("skip past EOF");
    }
    pos_ += n;
  }

  std::string read_string() {
    const std::uint64_t len = read<std::uint64_t>();
    if (len > static_cast<std::uint64_t>(size_ - pos_)) {
      throw std::runtime_error("string past EOF");
    }
    const char* p = reinterpret_cast<const char*>(data_ + pos_);
    std::string s(p, p + len);
    pos_ += static_cast<std::size_t>(len);
    return s;
  }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t pos_ = 0;
};

inline std::size_t align_up(std::size_t v, std::size_t alignment) {
  if (alignment == 0) {
    return v;
  }
  const std::size_t r = v % alignment;
  return r == 0 ? v : (v + (alignment - r));
}

}  // namespace cieft
