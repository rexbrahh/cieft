#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cieft {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("open failed: " + path);
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
      ::close(fd_);
      throw std::runtime_error("fstat failed: " + path);
    }
    if (st.st_size < 0) {
      ::close(fd_);
      throw std::runtime_error("invalid file size: " + path);
    }

    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ == 0) {
      ::close(fd_);
      throw std::runtime_error("empty file: " + path);
    }

    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("mmap failed: " + path);
    }

    data_ = static_cast<const std::uint8_t*>(mapped);

    // mapping is established; fd no longer needed
    ::close(fd_);
    fd_ = -1;
  }

  ~MappedFile() {
    if (data_ != nullptr && size_ != 0) {
      ::munmap(const_cast<std::uint8_t*>(data_), size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
  }

  MappedFile& operator=(MappedFile&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr && size_ != 0) {
      ::munmap(const_cast<std::uint8_t*>(data_), size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  const std::uint8_t* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace cieft
