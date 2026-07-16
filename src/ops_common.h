// shared graph helpers for the LIA/StyleGAN pieces (menc + mdec)
#pragma once

#include "tp_internal.h"

namespace tp {

// EqualConv2d-style conv (scale already folded into the weight at load).
// a [KW,KH,IC,OC], b [W,H,C,N]
inline ggml_tensor *conv2d(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                           int s, int p) {
    return ggml_conv_2d_direct(ctx, a, b, s, s, p, p, 1, 1);
}

// FusedLeakyReLU: (x + bias) -> leaky_relu(0.2) -> * sqrt(2)
// bias ggml ne [1,1,C,1] (from torch [1,C,1,1])
inline ggml_tensor *fused_lrelu(ggml_context *ctx, ggml_tensor *x, ggml_tensor *bias) {
    x = ggml_add(ctx, x, bias);
    x = ggml_leaky_relu(ctx, x, 0.2f, false);
    return ggml_scale(ctx, x, 1.41421356237309515f);
}

// upfirdn2d(up=1, down=1, pad=(pad0,pad1)) with a per-layer 4x4 FIR kernel
// stored as [4,4]; kernels are symmetric so no flip is needed.
// kern4 must be [KW,KH,1,C] (host-tiled per channel count).
inline ggml_tensor *blur(ggml_context *ctx, ggml_tensor *x, ggml_tensor *kern4,
                         int pad0, int pad1) {
    x = ggml_pad_ext(ctx, x, pad0, pad1, pad0, pad1, 0, 0, 0, 0);
    return ggml_conv_2d_dw_direct(ctx, kern4, x, 1, 1, 0, 0, 1, 1);
}

// zero-stuff x2 upsample (pixel at even coords): nearest x2 * checker mask.
// mask ne [2W, 2H, 1, 1] with 1 at (even,even), else 0.
inline ggml_tensor *zero_stuff2(ggml_context *ctx, ggml_tensor *x, ggml_tensor *mask) {
    ggml_tensor *up = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
    return ggml_mul(ctx, up, mask);
}

}  // namespace tp
