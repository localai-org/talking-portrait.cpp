# Round 2 summary — Vulkan f16 ship-config performance

Goal: push the Vulkan coopmat2-f16 ship configuration toward the
torch.compile fp16 numbers (pipeline 10 s RTF 16.47, decoder 513.3 fps,
peak VRAM <= 2946 MB) inside a 6-hour budget.

## Result

| metric | baseline | final | target | fraction |
|---|---|---|---|---|
| pipeline 10 s RTF | 6.57 | 10.76-10.85 | 16.47 | 65.9% |
| decoder fps | 192.1 | 355 (2.81 ms, batch 8) | 513.3 | 69.3% |
| peak VRAM | 2572 MB | 2690-2710 MB | <= 2946 | OK |

(final numbers = three consecutive passing full `gate-r2` runs at stop;
all improvements were banked through passing full gate runs, quality guard
green at every step.)

## What was done (chronological, each step gated)

1. **Cross-frame batching of the fast decoder graph** (biggest structural
   change). The whole fast graph now carries a batch dim: styles [512, B],
   CL activations [C, N, Bc], one graph dispatch decodes B=8 frames.
   Per-stage batch caps chunk the batch inside the graph (chunk lists +
   views, no concats) so peak im2col memory stays bounded; caps are
   monotone non-increasing across stages so chunk slices always align.
   The batched `im2col_cf` needed batch support in the CUDA + Vulkan
   kernels (src [IC,IW,IH,N]); `bilinear_gather` needed none (grid flattens
   across batch, feats are shared). mdec 5.19 -> 4.80 ms.
   - Stage bisection showed the win was concentrated in r8-r64 (launch
     bound: 1.32 -> 0.32 ms/frame); r64+ were GPU-work bound.

2. **Spatial direct-conv stages for r256/r512** (later r128 measured, then
   reverted to CL there — the fused CL subpix op made CL win again at r128).
   The big maps were dominated by im2col materialization traffic
   (~300 MB/frame at 512², reads at memory peak). Those stages now run
   spatially with implicit-GEMM `GGML_OP_CONV_2D`:
   - dense 3x3 conv: style/demod folded into the (tiny) per-frame conv
     weights — no im2col, no full-map modulate/demodulate passes;
   - up-conv: the composed sub-pixel kernel (conv_transpose ∘ blur, from
     round 1) as one 3x3 conv producing 4 parity classes, split into TWO
     half convs (see landmine below), followed by a fused pixel-shuffle;
   - ToFlow/ToRGB as 1x1 direct convs (much faster than m=3 GEMMs).
   mdec 4.80 -> 3.78 ms.
   - **New landmine found**: the Vulkan CONV_2D pipelines deterministically
     corrupt wide-Cout shapes here ([3,3,64,128], [3,3,128,256] — garbage
     regardless of weight dtype/provenance, map-side or folded, fusion and
     graph-optimize off; CPU backend clean, narrow-Cout shapes clean).
     Same family as the scalar-conv2d corruption in docs/vulkan-perf.md.
     Workaround: split the parity conv into 2 half-Cout convs.

3. **Custom fused pointwise ops** (all with CPU + CUDA + Vulkan kernels,
   patches/ggml-tp-ops.patch regenerated; ship graph = 100% Vulkan-native):
   - `GGML_OP_BIAS_LRELU` y = leaky(x+b) (axis 0 CL / axis 2 spatial);
   - `GGML_OP_SUBPIX_LRELU` 2x2 sub-pixel interleave of the two half-conv
     outputs + bias + leaky in one gather pass (spatial);
   - `GGML_OP_SUBPIX_LRELU_CL` same for the CL GEMM stages (single y4 src);
   - `GGML_OP_UPFIRDN2X` fused upfirdn2d(up=2, pad=(2,1), 4x4 FIR) for the
     rgb/flow skip pyramids (replaces upscale+mask-mul+pad+dw-conv);
   - `GGML_OP_CLSP_MUL` / `GGML_OP_CLSP_LERP` fused CL->spatial transpose
     with the warp mask (warp = T(sampled)*m, trunk = warp + out*(1-m));
   - `GGML_OP_FLOW_GRID` fused flow-map head (tanh+identity-grid x2,
     sigmoid) emitting a planar [N,3] grid|mask, consumed by a new planar
     mode of `bilinear_gather`.
   mdec 3.78 -> 3.11 -> 2.97 -> 2.86 ms across these, each step verified
   bit-consistent (PSNR ~68 dB vs the pre-change frames, i.e. fp16 noise).

4. **Direct frame write-out**: the batched decoder now writes full batches
   straight into the caller's frame buffer (`tp_generate` no longer bounces
   every frame through a per-frame callback memcpy; the padded tail batch
   still uses a bounce buffer). ~750 MB of host memcpy per 10 s clip gone:
   RTF 10.46 -> 10.83.

5. **Small wins**: flat-batch GEMMs when the A-matrix is shared (batched
   broadcast mul_mat re-reads A per batch element on Vulkan; folding the
   batch into N streams it once — conv1's n=16 GEMM went 312 -> ~40 us);
   graph-size scaling with batch; per-stage batch-cap/spatial-start tuning
   (ship: B=8, spatial from r256, caps 8,8,8,4,4,1,1 — mdec 2.82 ms,
   sampled peak ~2.19 GB).

## What did NOT work

- `bilinear_gather` writing the spatial layout directly (dst [N, C]):
  correct but slower — the CL-layout gather + separate transpose keeps
  feat reads warp-coalesced; the layout-direct write pattern loses more on
  scattered reads than it saves on the dropped transpose pass.
- conv_transpose_2d + separate blur for the spatial up-conv: works and is
  clean, but the implicit-GEMM conv_transpose burns 4x the essential MACs
  and the separate blur/pad passes cost ~1 ms/8fr more than the composed
  parity-class conv + fused pixel shuffle.
- Wide-Cout single-conv parity up-convs: driver corruption (above).
- B=16/32: no gain over B=8 (launch overhead already amortized).
- Spatial direct-conv at r128: within noise vs the batched CL GEMM path
  once SUBPIX_LRELU_CL landed; CL kept.

## Where the remaining ~1.9x lives (with more time)

- The decoder is ~2.8 ms/frame vs the 1.95 ms target. Per-op logs say the
  rest is: the r512 convs (M=32/64 shapes run at 29-54 TFLOPS on the
  Vulkan conv pipelines — a channels-last conv shader or direct use of the
  parity trick inside a custom conv would help), the bilinear gather
  (random-read bound), and ~1 ms/frame of remaining pointwise/copy passes
  that a real conv-epilogue fusion (bias+lrelu+warp inside the conv
  kernels, like inductor does) would absorb. Vulkan CONV_2D only accepts
  f32 activations, so f16 activation storage — the other half of the
  compiled reference's advantage — is blocked without backend surgery.
- Non-decoder time is ~220 ms per 10 s clip (ODE 5x21 ms, menc 18, audio
  12, GPU->host frame readback ~60). The wav2vec conv stem (2x slower than
  CUDA) and overlapping decode with the readback are the next levers.
- The batched exact graph idea (batching the conservative graph for
  parity-dump throughput) was out of scope; parity taps stay batch-1.

## Landmines / notes for future rounds

- Vulkan CONV_2D corrupts wide-Cout shapes on this driver (595.71.05):
  [3,3,64,128] and [3,3,128,256] garbage, [3,3,64,64] / [3,3,128,128] /
  [3,3,256,256] clean. Split convs on Cout to stay narrow.
- Batched-broadcast mul_mat (2D A, 3D B) re-reads A per batch element on
  Vulkan; flatten the batch into N when A is shared.
- All 7 new/extended ops follow the bilinear_gather pattern; the CUDA
  backend builds clean (docker recipe from README) and the CPU fallbacks
  are exercised by the debug knobs (TP_MDEC_NOFUSED=1 reverts the graph to
  the unfused ops; TP_MDEC_SPATIAL / TP_MDEC_BMAX / TP_MDEC_BATCH tune the
  structure; TP_MDEC_UPSP_TCONV / _CLCONV / _MAPSIDE keep the A/B paths).
- Quality: the fast-graph changes never touch the exact/tap graph, so
  mdec parity dumps are unchanged by construction; every optimization was
  additionally validated frame-level (PSNR 67-71 dB against the previous
  build's frames) before a full gate run.
