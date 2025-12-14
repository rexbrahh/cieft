#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gguf_loader.h"
#include "weights.h"

namespace {

struct Stats {
  std::size_t samples = 0;
  std::size_t nans = 0;
  std::size_t infs = 0;
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

Stats sample_stats(const float* data, std::uint64_t n, std::uint64_t max_samples = 1'000'000) {
  Stats s;
  if (n == 0) {
    s.min = 0;
    s.max = 0;
    return s;
  }
  const std::uint64_t step = std::max<std::uint64_t>(1, n / max_samples);
  for (std::uint64_t i = 0; i < n && s.samples < max_samples; i += step) {
    const float v = data[i];
    s.samples += 1;
    if (std::isnan(v)) {
      s.nans += 1;
      continue;
    }
    if (!std::isfinite(v)) {
      s.infs += 1;
      continue;
    }
    if (v < s.min) s.min = v;
    if (v > s.max) s.max = v;
  }
  if (!std::isfinite(s.min)) s.min = 0;
  if (!std::isfinite(s.max)) s.max = 0;
  return s;
}

void print_tensor_stats(std::string_view name, const cieft::TensorF32& t) {
  const auto st = sample_stats(t.data(), t.numel);
  std::cout << name << " dims=";
  std::cout << "[";
  for (std::size_t i = 0; i < t.dims.size(); i++) {
    if (i) std::cout << ", ";
    std::cout << t.dims[i];
  }
  std::cout << "]";
  std::cout << " samples=" << st.samples;
  std::cout << " nan=" << st.nans;
  std::cout << " inf=" << st.infs;
  std::cout << " min=" << std::setprecision(6) << st.min;
  std::cout << " max=" << std::setprecision(6) << st.max;
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << (argc > 0 ? argv[0] : "smoke_load")
                << " <model.gguf> [--layer N] [--lm-head]\n";
      return 2;
    }

    std::string path;
    std::uint32_t layer = 0;
    bool lm_head = false;

    path = argv[1];
    for (int i = 2; i < argc; i++) {
      const std::string_view a = argv[i];
      if (a == "--lm-head") {
        lm_head = true;
      } else if (a == "--layer") {
        if (i + 1 >= argc) {
          throw std::runtime_error("--layer requires an argument");
        }
        layer = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      } else {
        throw std::runtime_error("unknown arg: " + std::string(a));
      }
    }

    const cieft::GGUFLoader loader(path);
    const auto cfg = loader.config();

    std::cout << "config: n_layers=" << cfg.n_layers << " d_model=" << cfg.d_model << " n_heads=" << cfg.n_heads
              << " n_kv_heads=" << cfg.n_kv_heads << " head_dim=" << cfg.head_dim << " kv_dim=" << cfg.kv_dim
              << " ffn_hidden_dim=" << cfg.ffn_hidden_dim << " vocab=" << cfg.vocab_size << " rope_dim=" << cfg.rope_dim
              << " rope_theta=" << cfg.rope_theta << " rms_epsilon=" << cfg.rms_epsilon << "\n";

    auto weights = cieft::load_weights(loader, {layer}, lm_head);

    print_tensor_stats("token_embd.weight", weights.global.token_embd);

    if (weights.global.output_norm) {
      print_tensor_stats("output_norm.weight", *weights.global.output_norm);
    }
    if (weights.global.output) {
      print_tensor_stats("output.weight", *weights.global.output);
    }

    const auto& lw = weights.layers.at(0);
    print_tensor_stats("blk.attn_norm.weight", lw.attn_norm);
    print_tensor_stats("blk.attn_q.weight", lw.attn_q);
    print_tensor_stats("blk.attn_k.weight", lw.attn_k);
    print_tensor_stats("blk.attn_v.weight", lw.attn_v);
    print_tensor_stats("blk.attn_output.weight", lw.attn_output);
    print_tensor_stats("blk.ffn_norm.weight", lw.ffn_norm);
    print_tensor_stats("blk.ffn_gate.weight", lw.ffn_gate);
    print_tensor_stats("blk.ffn_up.weight", lw.ffn_up);
    print_tensor_stats("blk.ffn_down.weight", lw.ffn_down);

    // Quick embedding gather sanity check.
    std::vector<float> emb(cfg.d_model);
    cieft::gather_column(weights.global.token_embd, 1, emb.data());
    const auto emb_stats = sample_stats(emb.data(), emb.size(), emb.size());
    std::cout << "gather_column(token_embd.weight, token_id=1): min=" << emb_stats.min << " max=" << emb_stats.max
              << " nan=" << emb_stats.nans << " inf=" << emb_stats.infs << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

