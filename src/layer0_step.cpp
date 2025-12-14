#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "gguf_loader.h"
#include "layer0.h"
#include "weights.h"

namespace {

void print_head(const float* x, std::size_t n, std::size_t count = 16) {
  const std::size_t k = std::min(n, count);
  for (std::size_t i = 0; i < k; i++) {
    if (i) std::cout << " ";
    std::cout << std::setprecision(7) << x[i];
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << (argc > 0 ? argv[0] : "layer0_step")
                << " <model.gguf> --token <id> [--pos 0]\n";
      return 2;
    }

    std::string path = argv[1];
    std::uint32_t token = 0;
    bool have_token = false;
    std::uint32_t pos = 0;

    for (int i = 2; i < argc; i++) {
      const std::string_view a = argv[i];
      if (a == "--token") {
        if (i + 1 >= argc) throw std::runtime_error("--token requires an argument");
        token = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        have_token = true;
      } else if (a == "--pos") {
        if (i + 1 >= argc) throw std::runtime_error("--pos requires an argument");
        pos = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      } else {
        throw std::runtime_error("unknown arg: " + std::string(a));
      }
    }

    if (!have_token) {
      throw std::runtime_error("missing --token");
    }
    if (pos != 0) {
      throw std::runtime_error("this prototype currently supports only --pos 0 (single-token step)");
    }

    const cieft::GGUFLoader loader(path);
    auto weights = cieft::load_weights(loader, {0}, /*load_lm_head=*/false);

    if (token >= weights.cfg.vocab_size) {
      throw std::runtime_error("token id out of range for vocab");
    }

    std::vector<float> x(weights.cfg.d_model);
    cieft::gather_column(weights.global.token_embd, token, x.data());

    cieft::Layer0Context ctx(weights.cfg);
    ctx.step(weights.layers.at(0), pos, x.data());

    std::cout << "layer0 output (first 16 floats):\n";
    print_head(x.data(), x.size(), 16);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

