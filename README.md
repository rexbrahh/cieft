# cieft

Small, boring, bounds-checked GGUF tooling (macOS arm64).

What’s in here:

- `bin/inspect`: prints GGUF header/metadata + a full tensor map (dtype, shape, file offsets).
- `bin/smoke_load`: loads + dequantizes a small subset of tensors to float32 and prints basic stats.
- `bin/layer0_step`: a prototype “layer 0, one token” forward step for LLaMA-style (incl. GQA) models.
- `bin/two_layer_nn`: a tiny exercise program that prints every intermediate vector.
- `bin/two_layer_nn_sample`: a tiny exercise program that can greedy-pick from logits or sample with temperature.

`*.gguf` files are ignored by git; keep model files local.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCIEFT_MCPU=native
cmake --build build -j
```

Binaries are emitted into `bin/` (and ignored by git).

## Tools

### Inspect a GGUF

```sh
./bin/inspect path/to/model.gguf
```

Prints:

- header (version, tensor count, metadata count)
- selected metadata keys + `tokenizer.*` summaries
- dtype histogram
- all tensor entries (name, dtype, shape, absolute file offset, size in bytes)

### List RoPE/bias metadata keys

```sh
./scripts/check_RoPE_metadata_keys.py path/to/model.gguf
```

Optional substring filter(s):

```sh
./scripts/check_RoPE_metadata_keys.py path/to/model.gguf rope
```

### Smoke test: load + dequant (layer 0)

```sh
./bin/smoke_load path/to/model.gguf --layer 0
```

Add `--lm-head` to also load/dequant `output_norm.weight` and `output.weight`.

### Prototype: layer 0 single-token step

```sh
./bin/layer0_step path/to/model.gguf --token 1 --pos 0
```

Notes:

- currently supports only `--pos 0` (single token) in the prototype
- matrix weights are interpreted as `[in, out]` and applied as `y = W^T x` (columns contiguous)

## Exercises

### Two-layer NN (4 → 8 → 3)

```sh
./bin/two_layer_nn
./bin/two_layer_nn 0.1 -0.2 0.3 0.4
```

Prints: input, hidden pre-activation, hidden activation, logits, softmax, argmax.

### Two-layer NN sampling (temperature)

Greedy uses logits directly (`argmax(logits)`), softmax is only needed for sampling.

```sh
./bin/two_layer_nn_sample
./bin/two_layer_nn_sample --temperature 0.7 --seed 123
```
