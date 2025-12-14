#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cieft {

class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  static AlignedBuffer allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
      throw std::runtime_error("AlignedBuffer::allocate: bytes=0");
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw std::runtime_error("AlignedBuffer::allocate: alignment must be power of 2");
    }

    void* p = nullptr;
    const int rc = ::posix_memalign(&p, alignment, bytes);
    if (rc != 0 || p == nullptr) {
      throw std::runtime_error("posix_memalign failed");
    }
    return AlignedBuffer(p, bytes);
  }

  ~AlignedBuffer() {
    if (ptr_ != nullptr) {
      ::free(ptr_);
    }
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept { *this = std::move(other); }
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (ptr_ != nullptr) {
      ::free(ptr_);
    }
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    return *this;
  }

  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }

 private:
  AlignedBuffer(void* ptr, std::size_t bytes) : ptr_(ptr), bytes_(bytes) {}

  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace cieft

