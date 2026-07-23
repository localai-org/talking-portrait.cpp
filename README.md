# talking-portrait.cpp

C++/GGML+CUDA port of [FLOAT](https://github.com/deepbrainai-research/float)
(audio-driven talking portrait): one 512x512 portrait + 16 kHz speech ->
25 fps talking-head video. Reference harness: `/home/rich/c/float-harness`
(read-only; docs/TAPS.md is the frozen parity contract, `bin/gate` the only
pass/fail authority).

Pipeline: audio encoder (modified wav2vec2-base: group-norm conv stem,
temporal interp to 25 fps before the post-norm transformer, 12 layers
stacked -> Linear+LN+SiLU) -> flow-matching DiT (8 blocks, per-frame adaLN,
banded attention, 9 Euler steps, 3-way CFG per 50-frame chunk, 10-frame
autoregressive context) -> LIA/StyleGAN motion decoder per frame
(ModulatedConv2d pyramid, ToFlow grid_sample warping of encoder features,
ToRGB skip accumulation). Emotion is a caller-supplied one-hot (SER out of
scope).

## Layout

| path | what |
|---|---|
| `include/tp.h` | public C API (opaque handle, no exceptions across ABI, caller-owned buffers) |
| `src/` | subnets (`audio_enc`, `motion_enc`, `fmt`, `motion_dec`), pipeline, GGUF loader, tap dump writer, wav/image loaders |
| `tools/tp-cli` | portrait + wav + emotion -> frames.f32 / mp4 (ffmpeg) |
| `tools/tp-parity` | parity-dump runner for the harness gates (also `--generate` for M6 frames) |
| `bench/tp-bench` | per-stage + pipeline benchmark, emits the harness `bench_torch.py --json` row shape |
| `bench/tp-allocproof` | zero-allocation evidence (counting allocator + GPU pool flatness) |
| `tests/` | ctest suite (asset-gated tests SKIP with 77) |
| `fuzz/` | libFuzzer targets: wav, image, meta.json, CLI args |
| `ggml/` | vendored ggml submodule (ggml-org @ 3af5f576) |
| `patches/ggml-tp-ops.patch` | local ggml additions, auto-applied at configure (see below) |

## Build

Requirements: CMake >= 3.24, Ninja, a C++17 compiler, and git (the configure
step applies `patches/ggml-tp-ops.patch` to the ggml submodule on fresh
checkouts). `tp-cli --out *.mp4` shells out to `ffmpeg` at runtime.

```sh
git clone --recurse-submodules <repo-url> && cd talking-portrait.cpp
cmake --preset release
cmake --build --preset release -j
```

Presets: `debug` (ASan/UBSan), `release`, `cuda`, `vulkan`, `fuzz`
(clang+libFuzzer). The `vulkan` preset additionally needs Vulkan headers,
the loader and `glslc`.

Tests: `TP_GGUF_DIR=<harness>/ggufs TP_REF_DUMPS=<harness>/dumps ctest --test-dir build/release`

### Nix

`flake.nix` provides dev shells with the toolchains above:

```sh
nix develop           # CPU: cmake, ninja, gcc, git, ffmpeg
nix develop .#vulkan  # + shaderc (glslc), Vulkan headers/loader
nix develop .#fuzz    # clang stdenv for the fuzz preset
```

`nix build '.?submodules=1'` builds the release tools (`tp-cli`, `tp-parity`,
`tp-bench`, `tp-allocproof`) into `result/bin` (the `?submodules=1` is
required so the flake source includes the ggml submodule).

### CUDA (docker, sm_120)

```sh
docker run --rm --device nvidia.com/gpu=all -v $PWD:/work -w /work trellis2-demo:latest \
    bash -c 'cmake --preset cuda && cmake --build --preset cuda -j$(nproc)'
```

## ggml patch

`patches/ggml-tp-ops.patch` (applied automatically by CMake to the pristine
submodule checkout) adds two things needed for the M7 performance bar:

- `GGML_OP_BILINEAR_GATHER` (`ggml_bilinear_gather`): fused torch-style
  `grid_sample` (bilinear, zeros padding, `align_corners=false`) over a
  channels-first flattened feature map. CPU + CUDA + Vulkan. Replaces ~30 ops and 4
  `get_rows` per decoder stage.
- `ggml_im2col_cf`: im2col reading a channels-first `[C,W,H]` source with
  ic-fastest output rows, plus a block-per-pixel int32 CUDA kernel (the
  stock kernel's int64 index math dominated decode time) and a Vulkan
  workgroup-per-pixel shader.

Vulkan build + measurements: see `docs/vulkan-perf.md` (preset `vulkan`,
`bench-vulkan.json`).

## Numerics

- Parity dumps are true fp32: `tp-parity` sets `NVIDIA_TF32_OVERRIDE=0`
  (ggml enables cuBLAS TF32 by default; the fp32+TF32 mode is what the
  PyTorch baseline benches with, so `tp-bench` leaves it on).
- Equal* runtime scales (`1/sqrt(fan_in)`), the FusedLeakyReLU `sqrt(2)`,
  the ResBlock `1/sqrt(2)` and the QR-orthonormal Direction basis are folded
  at weight-load time; ModulatedConv2d demodulation uses precomputed
  per-conv Gram reductions (`S_aug`), halved to absorb the activation gain.
- The decoder has two graph builders: a numerically conservative spatial one
  (direct convolutions, map-side modulation) used in parity/tap mode where
  small-row cosine gates demand reference-order accumulation, and a fast
  channel-last one (sub-pixel composed conv_transpose+Blur, cf-im2col GEMMs,
  fused bilinear gather) used for generation — gated end-to-end by M6 SSIM.
- `--ftype f16` keeps the large FMT/decoder weights in f16 (audio + motion
  encoder stay f32); this is the benchmark configuration.

## Milestones

M1-M6 gates PASS at the default `cuda-f32` profile from CUDA-generated
dumps; M1-M4+M6 also pass at `cuda-f16` with `--ftype f16` (M5-f16 fails
only on the 2-element grid-pair cosine gate, inherent to f16 trunk noise).
M7: `bench.json` (RTF 7.0+ vs bar 6.86, mdec 213+ fps vs bar 190.5),
`alloc-evidence.txt` (0 heap allocs/frame on the generation thread, flat
GPU pool), `fuzz-evidence.txt` (4 targets, >4 CPU-hours).

## CLI

```sh
tp-cli --gguf <harness>/ggufs --image face.jpg --wav speech_16k.wav \
       --emotion happy --out out.mp4 [--ftype f16] [--device gpu|cpu]
```

## WebGL gallery demo

The local demo accepts a portrait plus either an uploaded audio file or a
browser microphone recording (up to 30 seconds). It generates a synchronized
MP4 in process through `libtp.so`, then plays that video as the texture of a
procedural picture frame in a WebGL2 gallery. Completed jobs and their original
inputs are retained in `generations/` until deleted in the UI.

```sh
scripts/demo.sh
# then open http://127.0.0.1:8743
```

The launcher defaults to `/home/rich/c/float-harness/ggufs`; override
`GGUF_DIR`, `STORE_DIR`, or `PORT` as needed. It builds both the CUDA shared
library and Go server inside the pinned demo container. The HTTP server binds
to localhost by default.

For a native build, configure with `-DTP_SHARED=ON` to produce `libtp.so`;
ordinary presets continue to build the original static library.

Recording is deliberately finite in this first version: press record, speak,
then stop to start inference. The current model encodes a complete clip before
it decodes frames, so continuous live-microphone animation requires a future
streaming inference API rather than a browser-only change.

## License

MIT (see `LICENSE`). The FLOAT weights this port runs are CC BY-NC —
weight-derived artifacts (GGUFs, dumps, generated video) must not be
redistributed; see `NOTICE.md`.
