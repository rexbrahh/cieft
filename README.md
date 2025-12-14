# cieft

Small, boring, bounds-checked GGUF tooling (macOS arm64).

What’s in here:

- `inspect`: prints GGUF header/metadata + a full tensor map (dtype, shape, file offsets).
- `smoke_load`: loads + dequantizes a small subset of tensors to float32 and prints basic stats.
- `layer0_step`: a prototype “layer 0, one token” forward step for LLaMA-style (incl. GQA) models.
- `two_layer_nn`: a tiny exercise program that prints every intermediate vector.

`*.gguf` files are ignored by git; keep model files local.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCIEFT_MCPU=native
cmake --build build -j
```

## Tools

### Inspect a GGUF

```sh
./inspect path/to/model.gguf
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
./smoke_load path/to/model.gguf --layer 0
```

Add `--lm-head` to also load/dequant `output_norm.weight` and `output.weight`.

### Prototype: layer 0 single-token step

```sh
./layer0_step path/to/model.gguf --token 1 --pos 0
```

Notes:

- currently supports only `--pos 0` (single token) in the prototype
- matrix weights are interpreted as `[in, out]` and applied as `y = W^T x` (columns contiguous)

## Exercises

### Two-layer NN (4 → 8 → 3)

```sh
./two_layer_nn
./two_layer_nn 0.1 -0.2 0.3 0.4
```

Prints: input, hidden pre-activation, hidden activation, logits, softmax, argmax.
