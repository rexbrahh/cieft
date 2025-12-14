#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

template <std::size_t N>
using Vec = std::array<float, N>;

template <std::size_t OUT, std::size_t IN>
using Mat = std::array<std::array<float, IN>, OUT>;  // row-major: OUT rows, IN cols

template <std::size_t N>
void print_vec(std::string_view label, const Vec<N>& v) {
  std::cout << label << " [" << N << "]: ";
  for (std::size_t i = 0; i < N; i++) {
    if (i) std::cout << " ";
    std::cout << std::setprecision(7) << v[i];
  }
  std::cout << "\n";
}

template <std::size_t OUT, std::size_t IN>
Vec<OUT> linear(const Mat<OUT, IN>& w, const Vec<IN>& x, const Vec<OUT>& b) {
  Vec<OUT> y{};
  for (std::size_t o = 0; o < OUT; o++) {
    double sum = static_cast<double>(b[o]);
    for (std::size_t i = 0; i < IN; i++) {
      sum += static_cast<double>(w[o][i]) * static_cast<double>(x[i]);
    }
    y[o] = static_cast<float>(sum);
  }
  return y;
}

template <std::size_t N>
Vec<N> relu(const Vec<N>& x) {
  Vec<N> y{};
  for (std::size_t i = 0; i < N; i++) {
    y[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }
  return y;
}

template <std::size_t N>
Vec<N> softmax(const Vec<N>& logits) {
  float max_v = logits[0];
  for (std::size_t i = 1; i < N; i++) {
    if (logits[i] > max_v) max_v = logits[i];
  }

  Vec<N> exps{};
  double sum = 0.0;
  for (std::size_t i = 0; i < N; i++) {
    const float e = std::exp(logits[i] - max_v);
    exps[i] = e;
    sum += e;
  }

  Vec<N> probs{};
  const float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
  for (std::size_t i = 0; i < N; i++) {
    probs[i] = exps[i] * inv_sum;
  }
  return probs;
}

template <std::size_t N>
std::size_t argmax(const Vec<N>& x) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < N; i++) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

template <std::size_t N>
std::size_t sample_categorical(const Vec<N>& probs, std::mt19937& rng) {
  // discrete_distribution accepts non-negative weights; we provide probs (already normalized).
  std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
  return dist(rng);
}

struct Options {
  Vec<4> x{0.10f, -0.20f, 0.30f, 0.40f};
  bool has_x = false;

  bool do_sample = false;
  float temperature = 1.0f;
  bool has_seed = false;
  std::uint32_t seed = 0;
};

Options parse_args(int argc, char** argv) {
  Options opt;
  std::vector<float> x_vals;

  for (int i = 1; i < argc; i++) {
    const std::string_view a = argv[i];
    if (a == "-h" || a == "--help") {
      std::cout << "usage: " << (argc > 0 ? argv[0] : "two_layer_nn_sample")
                << " [x0 x1 x2 x3] [--temperature T] [--seed S]\n"
                << "  - Greedy: argmax(logits)\n"
                << "  - Sampling: --temperature T (softmax(logits/T) then sample)\n";
      std::exit(0);
    }

    if (a == "--temperature") {
      if (i + 1 >= argc) throw std::runtime_error("--temperature requires a value");
      opt.temperature = std::stof(argv[++i]);
      opt.do_sample = true;
      continue;
    }
    if (a == "--seed") {
      if (i + 1 >= argc) throw std::runtime_error("--seed requires a value");
      opt.seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      opt.has_seed = true;
      continue;
    }

    // Positional inputs (x0..x3)
    if (x_vals.size() >= 4) {
      throw std::runtime_error("too many positional inputs (expected 4 floats)");
    }
    x_vals.push_back(std::stof(std::string(a)));
  }

  if (!x_vals.empty()) {
    if (x_vals.size() != 4) {
      throw std::runtime_error("expected exactly 4 floats for input x");
    }
    for (std::size_t i = 0; i < 4; i++) {
      opt.x[i] = x_vals[i];
    }
    opt.has_x = true;
  }

  if (opt.do_sample && !(opt.temperature > 0.0f)) {
    throw std::runtime_error("temperature must be > 0");
  }

  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Architecture:
    //   x (4) -> Linear (8) -> ReLU -> Linear (3) -> logits
    //   Greedy: argmax(logits)  (no softmax needed)
    //   Sample: softmax(logits / T) then sample

    const auto opt = parse_args(argc, argv);

    // Deterministic toy weights (same as two_layer_nn.cpp).
    Mat<8, 4> w1{};
    Vec<8> b1{};
    for (std::size_t h = 0; h < 8; h++) {
      b1[h] = static_cast<float>((static_cast<int>(h) - 3) * 0.05);
      for (std::size_t i = 0; i < 4; i++) {
        const float base = 0.05f * static_cast<float>((h + 1) * (i + 1));
        w1[h][i] = ((h + i) % 2 == 0) ? base : -base;
      }
    }

    Mat<3, 8> w2{};
    Vec<3> b2{};
    for (std::size_t o = 0; o < 3; o++) {
      b2[o] = static_cast<float>((static_cast<int>(o) - 1) * 0.1);
      for (std::size_t h = 0; h < 8; h++) {
        const float base = 0.03f * static_cast<float>((o + 1) * (h + 1));
        w2[o][h] = ((o + h) % 2 == 0) ? base : -base;
      }
    }

    const auto z1 = linear<8, 4>(w1, opt.x, b1);
    const auto h1 = relu<8>(z1);
    const auto logits = linear<3, 8>(w2, h1, b2);

    print_vec("x", opt.x);
    print_vec("z1 (hidden pre-activation)", z1);
    print_vec("h1 (hidden ReLU)", h1);
    print_vec("logits", logits);

    const auto greedy = argmax<3>(logits);
    std::cout << "greedy argmax(logits): " << greedy << "\n";

    if (opt.do_sample) {
      Vec<3> scaled{};
      for (std::size_t i = 0; i < 3; i++) {
        scaled[i] = logits[i] / opt.temperature;
      }
      const auto probs = softmax<3>(scaled);

      std::uint32_t seed = opt.seed;
      if (!opt.has_seed) {
        std::random_device rd;
        seed = rd();
      }
      std::mt19937 rng(seed);

      print_vec("scaled_logits (logits / T)", scaled);
      print_vec("softmax(scaled_logits)", probs);
      std::cout << "temperature: " << opt.temperature << "\n";
      std::cout << "seed: " << seed << "\n";

      const auto sampled = sample_categorical<3>(probs, rng);
      std::cout << "sampled: " << sampled << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

