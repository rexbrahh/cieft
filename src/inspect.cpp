#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <map>
#include <string>
#include <vector>

#include "gguf.h"
#include "mapped_file.h"

namespace {

std::string dims_to_string(const std::vector<std::uint64_t>& dims) {
  std::string out = "[";
  for (std::size_t i = 0; i < dims.size(); i++) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(dims[i]);
  }
  out += "]";
  return out;
}

const cieft::gguf::KV* find_kv(const cieft::gguf::File& f, const std::string& key) {
  auto it = f.kv_index_by_key.find(key);
  if (it == f.kv_index_by_key.end()) {
    return nullptr;
  }
  return &f.metadata[it->second];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      std::cerr << "usage: " << (argc > 0 ? argv[0] : "inspect") << " <model.gguf>\n";
      return argc == 2 ? 0 : 2;
    }

    const std::string path = argv[1];
    const cieft::MappedFile file(path);

    const auto gguf = cieft::gguf::parse(file.data(), file.size());

    std::cout << "A. Header\n";
    std::cout << "gguf version: " << gguf.header.version << "\n";
    std::cout << "number of tensors: " << gguf.header.tensor_count << "\n";
    std::cout << "number of metadata entries: " << gguf.header.metadata_kv_count << "\n";

    std::cout << "\nB. Key metadata you care about\n";
    const std::string keys[] = {
        "general.architecture",
        "llama.block_count",
        "llama.embedding_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
        "llama.context_length",
    };
    for (const auto& k : keys) {
      if (const auto* kv = find_kv(gguf, k)) {
        std::cout << k << ": " << cieft::gguf::value_to_string(kv->value) << "\n";
      }
    }

    // Vocab-related keys (tokenizer.*) can be big; print summaries.
    for (const auto& kv : gguf.metadata) {
      if (kv.key.rfind("tokenizer.", 0) == 0) {
        std::cout << kv.key << ": " << cieft::gguf::value_to_string(kv.value) << "\n";
      }
    }

    // Dtype histogram
    std::map<std::string, std::uint64_t> hist;
    for (const auto& t : gguf.tensors) {
      auto traits = cieft::gguf::ggml_type_traits(t.ggml_type);
      const std::string name = traits ? traits->name : ("UNKNOWN(" + std::to_string(t.ggml_type) + ")");
      hist[name] += 1;
    }

    std::cout << "\nDtype histogram\n";
    for (const auto& [dtype, count] : hist) {
      std::cout << dtype << ": " << count << " tensors\n";
    }

    // Fallback size map computed from offsets (works even for unknown ggml dtypes).
    std::vector<std::uint64_t> size_from_offsets(gguf.tensors.size(), 0);
    std::vector<std::size_t> sorted_idx(gguf.tensors.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](std::size_t a, std::size_t b) {
      return gguf.tensors[a].offset < gguf.tensors[b].offset;
    });
    for (std::size_t i = 0; i < sorted_idx.size(); i++) {
      const std::size_t cur = sorted_idx[i];
      const std::uint64_t cur_abs = static_cast<std::uint64_t>(gguf.data_section_offset) + gguf.tensors[cur].offset;
      std::uint64_t next_abs = static_cast<std::uint64_t>(file.size());
      if (i + 1 < sorted_idx.size()) {
        const std::size_t nxt = sorted_idx[i + 1];
        next_abs = static_cast<std::uint64_t>(gguf.data_section_offset) + gguf.tensors[nxt].offset;
      }
      size_from_offsets[cur] = (next_abs >= cur_abs) ? (next_abs - cur_abs) : 0;
    }

    std::cout << "\nC. All tensor entries\n";
    std::cout << "name | dtype | shape | file_offset | data_size_bytes\n";

    for (std::size_t i = 0; i < gguf.tensors.size(); i++) {
      const auto& t = gguf.tensors[i];
      const auto traits = cieft::gguf::ggml_type_traits(t.ggml_type);
      const std::string dtype = traits ? traits->name : ("UNKNOWN(" + std::to_string(t.ggml_type) + ")");
      const std::uint64_t abs_off = static_cast<std::uint64_t>(gguf.data_section_offset) + t.offset;
      const std::uint64_t bytes = cieft::gguf::tensor_nbytes(t).value_or(size_from_offsets[i]);

      std::cout << t.name << " | " << dtype << " | " << dims_to_string(t.dims) << " | " << abs_off << " | " << bytes
                << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
