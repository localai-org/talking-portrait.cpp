# GGML Vulkan backend evaluation (post-experiment)

Evaluation of this port on GGML's Vulkan backend on the experiment machine
(RTX 5070 Ti 16 GB, driver 595.71.05, Ryzen 9 7900, NixOS). Performed after
the timed porting experiment closed; nothing here touches the frozen gate
record (`bin/gate` was not run; all correctness numbers below come from
`float-harness/scripts/compare_taps.py` invoked directly with the
`vulkan-f32` / `vulkan-f16` profiles).

## Build recipe (NixOS)

```sh
nix-shell -p cmake ninja gcc vulkan-headers vulkan-loader shaderc pkg-config git \
    --run 'cmake --preset vulkan && cmake --build --preset vulkan -j$(nproc)'
```

- `vulkan` preset = Release + `TP_VULKAN=ON` (`GGML_VULKAN=ON`), build dir
  `build/vulkan`. glslc comes from `shaderc`; `find_package(Vulkan)`
  resolves through the nix-shell CMake prefix path.
- Runtime: `libvulkan.so.1` must be on `LD_LIBRARY_PATH` (rust-playbook §1):

```sh
export LD_LIBRARY_PATH="$(nix build nixpkgs#vulkan-loader --print-out-paths --no-link)/lib:/run/opengl-driver/lib"
```

- Device pick: ggml enumerates 2 Vulkan devices here (5070 Ti + Raphael
  iGPU; llvmpipe is not exposed through this loader stack). `backend_t::init`
  prefers the GPU-type device whose description contains "NVIDIA" and logs
  the pick (`tp: backend Vulkan0 on NVIDIA GeForce RTX 5070 Ti`). ggml
  reports `matrix cores: NV_coopmat2` on the 5070 Ti.

## What changed (all additive; CUDA/CPU paths untouched)

1. `TP_VULKAN` CMake option + `vulkan` preset.
2. NVIDIA-preferring device walk + device log in `backend_t::init`.
3. CPU-fallback scheduler path in `plan_t`: if the primary backend cannot
   run every node, the plan commits through `ggml_backend_sched({gpu,cpu})`
   instead of a single-backend gallocr plan; cf-im2col
   (`IM2COL` + `op_params[6]` bit 1) is treated as unsupported on backends
   without a channels-first kernel, since they would silently compute the
   channels-last layout. Fully-supported graphs keep the old path.
4. `TP_MDEC_EXACT=1` (force conservative decoder graph for generation) and
   `TP_FORCE_FALLBACK=1` (force custom ops onto the CPU fallback)
   measurement knobs; `tp-bench --rows` subset filter.
5. Bug fix surfaced by the fallback: the patched CPU `im2col` f16 path
   parsed the channels-first flag but ignored it in the micro-kernel
   (garbage frames when the f16 fast graph's cf-im2col fell back to CPU;
   never reachable before because CUDA has its own cf kernel and CPU-only
   mode is f32).
6. Stretch: Vulkan kernels for both custom ops in the vendored ggml
   (`bilinear_gather.comp`: thread per (c,n) output element, f32/f16-feat
   variants; `im2col_cf.comp`: workgroup per output pixel, threads loop
   R=IC*KW*KH, f32/f16 dst), wired through `ggml_vk_op_f32`, plus
   `GGML_VULKAN_CHECK_RESULTS` clone cases. `patches/ggml-tp-ops.patch`
   regenerated with all of it.

## Correctness (compare_taps, all 4 cases, 576 taps per profile)

Reference points: the CUDA dumps pass 576/576 at `cuda-f32` and 576/576 at
`cuda-f16` (same thresholds as the vulkan profiles).

| Vulkan configuration | vulkan-f32 | vulkan-f16 |
|---|---|---|
| **record**: scalar-f32 shader math + `TP_FORCE_FALLBACK=1` | **576/576** | **576/576** |
| scalar-f32 shader math, custom ops on Vulkan | 544/576 (a-short 203/235) | 544/576 (same taps) |
| default math (coopmat2 etc.), custom ops on Vulkan | 490/576 | 537/576 |

- **Record configuration** (the committed `parity-dumps-vulkan/` and
  `parity-dumps-vulkan-f16/`): scalar-f32 shaders
  (`GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`)
  with the custom ops on the CPU via the scheduler. Every tap of every case
  passes at both profiles — CUDA-equivalent parity. The three disables are
  the Vulkan analog of the parity `NVIDIA_TF32_OVERRIDE=0`: ggml-Vulkan's
  mat-mul **and CONV_2D** pipelines stage tiles through fp16 shared memory
  whenever coopmat2 is available, and `GGML_VK_DISABLE_F16` alone does not
  turn that off for CONV_2D.
- **Custom ops on Vulkan** (scalar math row 2): the fast graph — the graph
  that generation and benchmarks use — passes everywhere it is exercised
  (b-multichunk, c-tiny, d-portrait2: 341/341 per profile, final-image PSNR
  63–125 dB), which validates both new kernels in both dtypes. The 32
  failing taps are all a-short *conservative-graph* stage taps at
  r256/r512, hit by the known issue below.
- **Default math** (row 3; what the benchmarks run): every failure is a
  marginal miss on the tightest row-cosine gates — convs (worst
  `menc.conv7` cos 0.9985 vs gate 0.99996) and the outlier-heavy last
  wav2vec2 layer (`aud.w2v.l11` cos ≈ 0.97) — plus their downstream
  echoes; **every image PSNR gate passes** at both profiles. Cause: fp16
  tile staging in the coopmat2 mul_mm/conv2d pipelines.

### Known issue: scalar CONV_2D corruption in single-stream exact graph

With the **scalar-f32 conv2d pipelines** (coopmat2 disabled) and the whole
conservative graph running as **one Vulkan submission** (custom gather on
GPU, no scheduler splits), the first K=64 CONV_2D (r256 up-conv, weights
[3,3,128,64] on the zero-stuffed 259×259×128 input) deterministically
produces garbage: alternating 4-channel blocks of zeros
(`GGML_VULKAN_CHECK_RESULTS`: `avg_err=0.336`, first error at channel 4,
inputs verified correct). The same node with the same pipeline key is
correct in the same process/config when the scheduler splits the stream
(`TP_FORCE_FALLBACK=1`), correct under coopmat2 conv variants (default
math shows only ~1e-4 cosine noise there), and unaffected by
`GGML_VK_DISABLE_GRAPH_OPTIMIZE`, `GGML_VK_DISABLE_FUSION`, and the driver
shader cache. Looks like a latent ggml-Vulkan scalar-conv2d/driver
interaction uncovered by the new dispatch sequence; parity therefore uses
the record configuration above. Not hit by the fast graph (no CONV_2D
there) — i.e. not hit by any benchmark configuration.

## Configuration matrix (all measured; bench-vulkan.json `configs`)

| config | decoder graph | custom ops | shader math | mdec | pipeline 10 s |
|---|---|---|---|---|---|
| **fast-f16-gpu** (headline) | fast | Vulkan | default | **5.17 ms / 193.4 fps** | **1515 ms / RTF 6.60** |
| fast-f32-gpu | fast | Vulkan | default | 7.00 ms / 143.0 fps | 1952 ms / RTF 5.12 |
| exact-f16-gpu | conservative | Vulkan | default | 6.56 ms / 152.6 fps | 1869 ms / RTF 5.35 |
| fast-f16-strict | fast | Vulkan | scalar f32 | 9.05 ms / 110.5 fps | (not run) |
| fast-f16-fallback (baseline) | fast | CPU (sched) | default | 2845 ms / 0.4 fps | RTF 0.01 (2 s row) |
| exact-f16 (baseline) | conservative | CPU gather | default | 1957 ms / 0.5 fps | 508.7 s / RTF 0.02 |
| exact-f32 (baseline) | conservative | CPU gather | default | 1581 ms / 0.6 fps | 449.1 s / RTF 0.02 |

Steady-state protocol: tp-bench warms up every stage before timing (Vulkan
pipeline compilation lands in the warmups); pipeline rows are a full
warmup generation + a timed generation; peak VRAM is sampled from
`nvidia-smi` at 200 ms (`scripts/bench-vulkan.sh`), with ggml's
device-level number kept alongside in the per-config JSON.

Per-stage rows of the headline config (vs CUDA `bench.json`):

| row | GGML-Vulkan f16 | GGML-CUDA f16 |
|---|---|---|
| menc (per portrait) | 18.2 ms | 14.3 ms |
| audio enc (per 10 s) | 12.1 ms | 6.6 ms |
| FMT fwd (CFG batch 3) | **2.35 ms** | 2.61 ms |
| ODE chunk (9 steps, 50 fr) | **21.3 ms** | 23.7 ms |
| mdec (per frame) | 5.17 ms | 4.69 ms |
| pipeline 2 s video | 357 ms (RTF 5.60) | 351 ms (RTF 5.69) |
| pipeline 10 s video | 1515 ms (RTF 6.60) | 1413 ms (RTF 7.08) |
| peak VRAM | 2110 MB sampled / 2572 dev | 2143 MB |

## Comparison table (10 s clip, f16 configurations)

| | GGML-Vulkan (this) | GGML-CUDA (recorded) | Rust burn-Vulkan (recorded) | PyTorch bar |
|---|---|---|---|---|
| mdec per frame | 5.17 ms / 193.4 fps (batch 1) | 4.69 ms / 213.2 fps | 4.76 ms / 210.0 fps (batch 8); 6.00 ms / 166.6 fps (batch 1) | 5.25 ms / 190.5 fps |
| pipeline 10 s | RTF 6.60, 165.0 fps | RTF 7.08, 176.9 fps | RTF 7.27, 181.9 fps | RTF 6.86, 171.6 fps |
| peak VRAM | 2110 MB | 2143 MB | 2821 MB | 10173 MB |

## Analysis

- **Transformer (attention + GEMM): GGML-Vulkan matches or beats CUDA.**
  FMT forward 2.35 vs 2.61 ms and the 9-step ODE chunk 21.3 vs 23.7 ms —
  NV_coopmat2 mat-muls plus fp16 mul_mm staging make the DiT side fully
  competitive; there is no Vulkan penalty on this model's banded attention.
- **Convolutions are the gap.** The wav2vec2 conv stem (audio enc 12.1 vs
  6.6 ms) and the StyleGAN encoder (menc 18.2 vs 14.3 ms) trail CUDA
  1.3–1.8×; the decoder's GEMM-formulated convs (fast graph) hold up
  better — mdec 5.17 vs 4.69 ms, within 10%.
- **The custom-op gap is decisive, not incremental.** Without Vulkan
  kernels for the two patch ops, the honest scheduler-fallback baseline is
  RTF 0.01–0.03 (0.4–0.6 fps decode): per-frame GPU→CPU→GPU round-trips of
  feature maps (~65 MB/frame exact, ~500 MB/frame fast) swamp everything —
  a ~300× end-to-end penalty. Two small shaders (~120 lines of GLSL plus
  dispatch plumbing) recover it entirely: 0.4 → 193 fps decode. On the
  GGML scheduler, "it falls back to CPU" is functionally correct but
  performance-fatal for per-frame vision ops.
- **Numerics tradeoff is real but bounded.** Default Vulkan math loses the
  strictest row-cosine parity gates (fp16 tile staging in mul_mm and
  conv2d) while passing every image gate; bit-honest f32 costs ~2×
  (fast-f16-strict: FMT 5.67 ms, ODE 51.4 ms, mdec 9.05 ms) and passes
  everything the CUDA port passes. CUDA's parity runs pin TF32 off for the
  same reason, but CUDA fp32 GEMMs stay fast; Vulkan currently has no
  tensor-core-f32 middle ground on this backend, so strict parity and
  peak speed are further apart than on CUDA.
- **Vs the Rust burn/Vulkan port**: GGML-Vulkan wins batch-1 decode
  (5.17 vs 6.00 ms) and peak VRAM (2.1 vs 2.8 GB); burn's recorded RTF
  7.27 leans on batch-8 decode (210 fps), which this port does not
  implement (frame-by-frame decode). End-to-end GGML-Vulkan lands at RTF
  6.60 — ~7% behind CUDA (7.08), ~4% below the PyTorch bar (6.86) — while
  its decode fps (193.4) clears the bar (190.5).
- **VRAM**: all three ports sit in the 2–3 GB envelope, ~4–5× under the
  PyTorch reference (10.2 GB).

## Reproduce

```sh
# parity (record configuration)
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
TP_FORCE_FALLBACK=1 ./build/vulkan/tp-parity --gguf <harness>/ggufs \
    --ref <harness>/dumps --out parity-dumps-vulkan --case a-short --device gpu
<harness>/bin/refrun python <harness>/scripts/compare_taps.py \
    <harness>/dumps/a-short parity-dumps-vulkan/a-short --profile vulkan-f32

# benchmarks
scripts/bench-vulkan.sh fast-f16-gpu f16                      # headline
TP_MDEC_EXACT=1     scripts/bench-vulkan.sh exact-f16-gpu f16 # conservative graph
TP_FORCE_FALLBACK=1 TP_BENCH_ARGS="--rows mdec,pipe2" \
    scripts/bench-vulkan.sh fast-f16-fallback f16             # no-kernel baseline
python3 scripts/assemble-bench-vulkan.py bench-vulkan.json
```
