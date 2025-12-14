#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
void print_mat(std::string_view label, const Mat<OUT, IN>& m) {
  std::cout << label << " [" << OUT << "x" << IN << "]\n";
  for (std::size_t o = 0; o < OUT; o++) {
    std::cout << "  ";
    for (std::size_t i = 0; i < IN; i++) {
      if (i) std::cout << " ";
      std::cout << std::setprecision(7) << m[o][i];
    }
    std::cout << "\n";
  }
}

template <std::size_t N>
float dot(const Vec<N>& a, const Vec<N>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < N; i++) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
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

struct Options {
  bool use_scale = true;
  Vec<4> x0{0.10f, -0.20f, 0.00f, 0.30f};
  Vec<4> x1{-0.10f, 0.40f, 0.20f, -0.30f};
};

Options parse_args(int argc, char** argv) {
  Options opt;
  std::vector<float> vals;

  for (int i = 1; i < argc; i++) {
    const std::string_view a = argv[i];
    if (a == "-h" || a == "--help") {
      std::cout << "usage: " << (argc > 0 ? argv[0] : "two_token_attention")
                << " [x0_0 x0_1 x0_2 x0_3 x1_0 x1_1 x1_2 x1_3] [--no-scale]\n"
                << "  - Computes Q,K,V for two tokens (dim=4)\n"
                << "  - Attention scores: score[i,j] = dot(q_i, k_j) / sqrt(d)\n"
                << "  - Attention weights: softmax over j\n"
                << "  - Output: out_i = sum_j w[i,j] * v_j\n";
      std::exit(0);
    }
    if (a == "--no-scale") {
      opt.use_scale = false;
      continue;
    }

    vals.push_back(std::stof(std::string(a)));
  }

  if (!vals.empty()) {
    if (vals.size() != 8) {
      throw std::runtime_error("expected exactly 8 positional floats: 4 for token0, 4 for token1");
    }
    for (std::size_t i = 0; i < 4; i++) opt.x0[i] = vals[i];
    for (std::size_t i = 0; i < 4; i++) opt.x1[i] = vals[4 + i];
  }

  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    constexpr std::size_t d = 4;
    constexpr std::size_t n_tok = 2;

    const Options opt = parse_args(argc, argv);

    // Two tokens, each dim=4.
    const std::array<Vec<d>, n_tok> x = {opt.x0, opt.x1};

    // Tiny deterministic weights for Q/K/V projections (dim=4 -> dim=4).
    Mat<d, d> wq{};
    Mat<d, d> wk{};
    Mat<d, d> wv{};
    Vec<d> bq{};
    Vec<d> bk{};
    Vec<d> bv{};

    for (std::size_t o = 0; o < d; o++) {
      bq[o] = static_cast<float>((static_cast<int>(o) - 2) * 0.01);
      bk[o] = static_cast<float>((static_cast<int>(o) - 1) * 0.02);
      bv[o] = static_cast<float>((static_cast<int>(o) - 0) * 0.015);

      for (std::size_t i = 0; i < d; i++) {
        const float base_q = 0.04f * static_cast<float>((o + 1) * (i + 1));
        const float base_k = 0.03f * static_cast<float>((o + 1) * (i + 2));
        const float base_v = 0.02f * static_cast<float>((o + 2) * (i + 1));

        wq[o][i] = ((o + i) % 2 == 0) ? base_q : -base_q;
        wk[o][i] = ((o + 2 * i) % 2 == 0) ? base_k : -base_k;
        wv[o][i] = ((2 * o + i) % 2 == 0) ? base_v : -base_v;
      }
    }

    print_vec("x0", x[0]);
    print_vec("x1", x[1]);
    std::cout << "\n";

    print_mat("Wq", wq);
    print_mat("Wk", wk);
    print_mat("Wv", wv);
    print_vec("bq", bq);
    print_vec("bk", bk);
    print_vec("bv", bv);
    std::cout << "\n";

    // Q, K, V per token.
    std::array<Vec<d>, n_tok> q{};
    std::array<Vec<d>, n_tok> k{};
    std::array<Vec<d>, n_tok> v{};
    for (std::size_t t = 0; t < n_tok; t++) {
      q[t] = linear<d, d>(wq, x[t], bq);
      k[t] = linear<d, d>(wk, x[t], bk);
      v[t] = linear<d, d>(wv, x[t], bv);
    }

    print_vec("q0", q[0]);
    print_vec("q1", q[1]);
    print_vec("k0", k[0]);
    print_vec("k1", k[1]);
    print_vec("v0", v[0]);
    print_vec("v1", v[1]);
    std::cout << "\n";

    // Attention scores and weights (2x2): for each query token i, softmax over keys j.
    const float scale = opt.use_scale ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(d))) : 1.0f;

    std::array<Vec<n_tok>, n_tok> scores{};
    std::array<Vec<n_tok>, n_tok> weights{};
    for (std::size_t i = 0; i < n_tok; i++) {
      for (std::size_t j = 0; j < n_tok; j++) {
        scores[i][j] = dot<d>(q[i], k[j]) * scale;
      }
      weights[i] = softmax<n_tok>(scores[i]);
    }

    print_vec("scores[0,*] (q0·k0, q0·k1)", scores[0]);
    print_vec("scores[1,*] (q1·k0, q1·k1)", scores[1]);
    std::cout << "scale: " << (opt.use_scale ? "1/sqrt(d)" : "1") << "\n\n";

    print_vec("attn_weights[0,*]", weights[0]);
    print_vec("attn_weights[1,*]", weights[1]);
    std::cout << "\n";

    // Mix values: out_i = sum_j w[i,j] * v_j
    std::array<Vec<d>, n_tok> out{};
    for (std::size_t i = 0; i < n_tok; i++) {
      Vec<d> y{};
      for (std::size_t j = 0; j < n_tok; j++) {
        for (std::size_t c = 0; c < d; c++) {
          y[c] += weights[i][j] * v[j][c];
        }
      }
      out[i] = y;
    }

    print_vec("out0 (mixed values)", out[0]);
    print_vec("out1 (mixed values)", out[1]);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

