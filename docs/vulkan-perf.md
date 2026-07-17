# GGML Vulkan backend evaluation (post-experiment)

Evaluation of this port on GGML's Vulkan backend, on the experiment machine
(RTX 5070 Ti 16 GB, driver 595.71.05, Ryzen 9 7900, NixOS). Performed after
the timed porting experiment closed; nothing here touches the frozen gate
record (`bin/gate` was not run, comparisons below use
`float-harness/scripts/compare_taps.py` directly).

## Build recipe (NixOS)

```sh
nix-shell -p cmake ninja gcc vulkan-headers vulkan-loader shaderc pkg-config git \
    --run 'cmake --preset vulkan && cmake --build --preset vulkan -j$(nproc)'
```

- `vulkan` preset = Release + `TP_VULKAN=ON` (`GGML_VULKAN=ON`), build dir
  `build/vulkan`. glslc comes from `shaderc`; `find_package(Vulkan)` resolves
  through the nix-shell CMake prefix path.
- Runtime: `libvulkan.so.1` must be on `LD_LIBRARY_PATH` (playbook §1):

```sh
export LD_LIBRARY_PATH="$(nix build nixpkgs#vulkan-loader --print-out-paths --no-link)/lib:/run/opengl-driver/lib"
```

- Device pick: ggml enumerates 2 Vulkan devices here (5070 Ti + Raphael
  iGPU; llvmpipe is not exposed through this loader stack). `backend_t::init`
  now prefers the GPU-type device whose description contains "NVIDIA" and
  logs the pick (`tp: backend Vulkan0 on NVIDIA GeForce RTX 5070 Ti`).
  ggml reports `matrix cores: NV_coopmat2` on the 5070 Ti.

## What had to change (all additive)

1. `TP_VULKAN` CMake option + `vulkan` preset.
2. NVIDIA-preferring device walk + device log in `backend_t::init`.
3. CPU-fallback scheduler path in `plan_t`: if the primary backend cannot
   run every node (`ggml_backend_supports_op`, plus a special case treating
   cf-im2col as unsupported on backends without a channels-first kernel),
   the plan commits through `ggml_backend_sched({gpu,cpu})` instead of a
   single-backend gallocr plan. CUDA/CPU builds are unaffected (their graphs
   are fully supported, so they keep the old path).
4. `TP_MDEC_EXACT=1` env knob to force the conservative decoder graph for
   generation (measurement aid).
5. Bug fix exposed by the fallback: the patched CPU `im2col` f16 path parsed
   the channels-first flag but ignored it in the micro-kernel, producing
   garbage frames (PSNR ~3 dB) when the f16 fast graph's cf-im2col fell back
   to the CPU. Never hit before (CUDA has its own cf kernel; CPU-only runs
   are f32). Fixed in the vendored ggml + regenerated patch.
6. Stretch (see below): Vulkan kernels for the two custom ops
   (`GGML_OP_BILINEAR_GATHER`, cf-im2col) in the vendored ggml.

## Correctness (compare_taps, all 4 cases)

Reference: the CUDA dumps pass 576/576 taps at `cuda-f32` and 576/576 at
`cuda-f16` (same comparator, same thresholds as the vulkan profiles).

Failed taps out of 576 (a-short 235 / b-multichunk 194 / c-tiny 75 /
d-portrait2 72):

| shader math | vulkan-f32 | vulkan-f16 |
|---|---|---|
| default (NV_coopmat2 GEMM+conv, fp16 shmem staging) | 84 fail (59/16/7/2) | TBD-after-fix (27/-/-/-) |
| `GGML_VK_DISABLE_F16=1` | 49 fail (44/0/4/1) | — |
| scalar f32 (`GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`) | **0 fail** | **0 fail** |

- All default-math failures are marginal cosine misses on the tightest
  row-cosine gates (worst `menc.conv7` cos 0.9985 vs gate 0.99996; image
  PSNR gates all pass even with default math). Cause: ggml-Vulkan's mat-mul
  and CONV_2D pipelines stage tiles through **fp16 shared memory whenever
  coopmat2 is available** — `GGML_VK_DISABLE_F16` alone does not turn this
  off for CONV_2D, only for mul_mat.
- With scalar f32 shaders the port passes every tap of every case at both
  profiles, same as CUDA. True-f32 parity on this backend therefore requires
  all three disables; that is the Vulkan analog of `NVIDIA_TF32_OVERRIDE=0`.
- The parity dumps in `parity-dumps-vulkan/` (f32) and
  `parity-dumps-vulkan-f16/` were generated with scalar math.

## Configuration matrix (what was measured)

| config | decoder graph | custom ops on | notes |
|---|---|---|---|
| baseline fast | fast (channel-last) | CPU via sched fallback (7 gather + 14 cf-im2col per frame) | honest baseline before Vulkan kernels |
| baseline exact | conservative (`TP_MDEC_EXACT=1`) | CPU fallback (7 gather per frame) | direct convs stay on Vulkan |
| stretch fast | fast | Vulkan (new kernels) | fully on-GPU |
| stretch exact | conservative | Vulkan (new kernels) | fully on-GPU |

## Benchmarks

Driver: `scripts/bench-vulkan.sh` (tp-bench + 200 ms `nvidia-smi` VRAM
sampling), default Vulkan shader math unless noted. Steady-state: every row
has warmup runs before timing; Vulkan pipeline compilation happens in the
warmups.

### Baseline (custom ops on CPU via scheduler fallback)

exact graph (`TP_MDEC_EXACT=1`), full bench rows:

| row | f16 | f32 |
|---|---|---|
| menc (per portrait) | 35.2 ms | 18.1 ms |
| audio enc (per 10 s) | 12.1 ms | 12.0 ms |
| FMT fwd (CFG batch 3) | 2.36 ms | 2.85 ms |
| ODE chunk (9 steps, 50 fr) | 21.3 ms | 25.9 ms |
| mdec (per frame) | 1957 ms (0.5 fps) | 1581 ms (0.6 fps) |
| pipeline 10 s | 508.7 s (RTF 0.02) | 449.1 s (RTF 0.02) |
| peak VRAM (nvidia-smi) | 2069 MB | 2364 MB |

fast graph baseline (7 gather + 14 cf-im2col fallbacks/frame): ~2.4 s/frame
(30-frame c-tiny clip, 81 s wall) — see matrix rows below for the
post-kernel re-measurement with `TP_FORCE_FALLBACK=1`.

The FMT/ODE rows are the striking part: on the transformer, GGML-Vulkan is
already at CUDA speed (CUDA: 2.61 ms FMT, 23.7 ms ODE). Everything decoder-
side collapses because each frame round-trips feature maps through the CPU
for the missing custom ops.

## Analysis

TBD
