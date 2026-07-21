# Round 3 summary — CPU ship-config performance

Goal: make the C++/GGML FLOAT port fast on **CPU** (Ryzen 9 7900, 12c/24t)
toward the torch.compile fp32 numbers (pipeline 10 s RTF 0.22, decoder
5.5 fps sustained) inside a 3-hour budget. The port was correct on CPU but
completely untuned.

## Result

| metric | baseline | final | target | fraction |
|---|---|---|---|---|
| pipeline 10 s RTF | 0.03 | 0.17 | 0.22 | ~77% |
| decoder fps (batch 8) | 0.9 | 4.8 (≈209 ms/frame) | 5.5 | ~87% |
| peak RSS | — | ~3.6 GB | (no cap) | OK |

Baseline was the operator's `build/vulkan` run. Final numbers are from three
consecutive passing full `gate-cpu --final` runs at stop; every improvement
was banked through a passing full gate, quality guard (mdec cpu-f32 taps
108/108, video SSIM on all 3 cases) green at every step. The frozen parity
exception list stayed EMPTY.

## What was done (chronological, each banked by a full gate)

1. **Build config — the big structural find (≈3.4×).** The operator's
   baseline ran `build/vulkan/tp-bench --device cpu`, and that build has
   `GGML_NATIVE=OFF` → no `-march=native` → ggml-cpu falls back to SSE/scalar
   kernels. The `release` preset has `GGML_NATIVE=ON` → AVX512 on Zen4.
   Pointing the R3 bench script at `build/release` alone took mdec
   1166 → ~350 ms/frame. Added a `TP_NTHREADS` env knob and pinned **12
   threads** (physical cores; 24-way SMT contends on the AVX512 GEMMs:
   12t 317 ms vs 24t 329 ms). mdec ~350 → 317 ms.

2. **Vectorized the fused activation kernels (bit-exact).** The custom CPU
   kernels for `bias_lrelu` / `subpix_lrelu` / `subpix_lrelu_cl` did a
   per-element integer division to recover the channel index. Restructured
   each to hoist the idiv out of the inner loop and make the inner loop a
   contiguous add + leaky-relu that `-O3 -march=native` autovectorizes.
   Same per-element math → bit-exact. mdec 317 → ~305 ms.

3. **Merged the CPU up-conv (bit-exact).** `styled_conv_up_sp` splits the
   up-conv into two half-Cout convs purely to dodge a Vulkan driver bug on
   wide-Cout `CONV_2D`. On CPU the wide conv is fine, so it now runs ONE
   `4·OC` conv and views the two row-parity halves — im2col is materialized
   once, not twice. mdec ~305 → 293 ms.

4. **Enabled `GGML_LLAMAFILE` — the single biggest lever (≈1.4×).** ggml's
   default f32 `mul_mat` runs the naive `vec_dot` path (≈3.5% of AVX512 peak
   for the decoder's skinny GEMM shapes). `GGML_LLAMAFILE=ON` routes f32/f16
   matmuls through llamafile's blocked tinyBLAS microkernels, which
   accelerates every matmul — including the ones *inside* `conv_2d_direct`
   and the FMT/ODE transformer. mdec 293 → 211 ms (4.8 fps); ODE chunk
   1050 → 599 ms; FMT 116 → 67 ms; pipeline RTF 0.11 → 0.17. mdec taps
   stayed 108/108 (err ~3e-5, gate 1e-2). Added to the `release` preset.

5. **De-idiv'd `clsp_mul`/`clsp_lerp` (bit-exact).** Marginal (the transpose
   is memory-bound), kept for a fully idiv-free fast-graph op set.

## What was measured but did NOT help

- **f16 weights**: slower than f32 both before (347 vs 317) and after
  llamafile (214 vs 209) — the on-the-fly conversion overhead outweighs the
  bandwidth win on CPU. Kept f32 (also the parity-safe choice).
- **Thread count ≠ 12**: 8t/10t/16t/24t all worse; 12 (= physical cores) is
  the sweet spot for the AVX512/llamafile GEMMs.
- **Batch size** (8/16/32) and **per-stage batch caps**: no effect once the
  weight is streamed once per GEMM.
- **Conv-strategy crossover** (`TP_MDEC_SPATIAL`): the shipped default
  (spatial direct conv from stage 5, CL im2col+GEMM below) is optimal; im2col
  everywhere (=7) is 20-25% slower because the custom `im2col_cf` is scalar.
- **Larger `GGML_IM2COL_WORK_SIZE`** (16→128 MB): the conv is not
  barrier/re-entry bound, no change.

## Where the time goes at the end (mdec ≈209 ms, 12t, f32)

Per-stage incremental cost: r8-r128 ≈43 ms, r256 ≈48 ms, **r512 ≈88 ms**.
mdec is ~92% of the pipeline. With llamafile the GEMMs are fast, so the
high-res stages are now dominated by **im2col materialization + the strided
permute-back in `conv_2d_direct`** (the r512 3×3 conv materializes a ~300 MB
im2col buffer). The custom warp/gather ops (`bilinear_gather`, `clsp`,
`flow_grid`) are already contiguous/vectorizable and are not the bottleneck.

## What I'd do next (didn't fit / too risky for the budget)

- **im2col-free 3×3 conv at high res**: decompose the same-res 3×3 into 9
  shifted 1×1 `mul_mat`s (llamafile-fast) accumulated, skipping the ~300 MB
  im2col write and the strided permute. Biggest remaining lever (r512 is
  41% of mdec); left out because a correct, layout-juggling implementation
  with SSIM re-verification was too risky in the remaining time.
- **Winograd F(2,3)** for the 3×3 convs (2.25× fewer mults) — the path
  torch's oneDNN takes; high parity risk against the frozen empty exception
  list.
- **Cache-blocked permute** in `conv_2d_direct` (bit-exact reorder) to kill
  the strided channel writes at high res.
