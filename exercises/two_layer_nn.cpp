#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

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

bool parse_input(int argc, char** argv, Vec<4>* out) {
  if (argc == 5) {
    for (int i = 0; i < 4; i++) {
      try {
        (*out)[i] = std::stof(argv[i + 1]);
      } catch (...) {
        return false;
      }
    }
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  // Architecture:
  //   x (4) -> Linear (8) -> ReLU -> Linear (3) -> Softmax -> Argmax

  Vec<4> x{0.10f, -0.20f, 0.30f, 0.40f};
  if (argc != 1 && !parse_input(argc, argv, &x)) {
    std::cerr << "usage: " << (argc > 0 ? argv[0] : "two_layer_nn") << " <x0> <x1> <x2> <x3>\n";
    std::cerr << "or run with no args for the default input.\n";
    return 2;
  }

  // Deterministic toy weights (just for demonstration).
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

  const auto z1 = linear<8, 4>(w1, x, b1);
  const auto h1 = relu<8>(z1);
  const auto logits = linear<3, 8>(w2, h1, b2);
  const auto probs = softmax<3>(logits);
  const auto pred = argmax<3>(probs);

  print_vec("x", x);
  print_vec("z1 (hidden pre-activation)", z1);
  print_vec("h1 (hidden ReLU)", h1);
  print_vec("logits", logits);
  print_vec("softmax", probs);
  std::cout << "argmax: " << pred << "\n";

  return 0;
}

