// shared graph helpers for the LIA/StyleGAN pieces (menc + mdec)
#pragma once

#include "tp_internal.h"

namespace tp {

// EqualConv2d-style conv (scale already folded into the weight at load).
// a [KW,KH,IC,OC], b [W,H,C,N]. im2col+GEMM: the CUDA direct-conv kernel is
// a naive per-element loop (4-14x slower); im2col dtype follows the weight
// (f16 weights -> f16 im2col -> tensor-core GEMM).
inline ggml_tensor *conv2d(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                           int s, int p) {
    ggml_tensor *im = ggml_im2col(ctx, a, b, s, s, p, p, 1, 1, true, a->type);
    ggml_tensor *am = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]);
    ggml_tensor *r = ggml_mul_mat(
        ctx, am, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]));
    r = ggml_reshape_3d(ctx, r, a->ne[3], im->ne[1], im->ne[2]);  // [OC, OW, OH]
    return ggml_cont(ctx, ggml_permute(ctx, r, 2, 0, 1, 3));      // [OW, OH, OC]
}

// FusedLeakyReLU: sqrt(2)*leaky(x+b). leaky is positively homogeneous, so
// the sqrt(2) is FOLDED INTO the upstream weights/bias at load (see
// motion_ae_scale / motion_dec::prepare) and this helper only adds + leaks.
// bias ggml ne [1,1,C,1] (from torch [1,C,1,1])
inline ggml_tensor *fused_lrelu(ggml_context *ctx, ggml_tensor *x, ggml_tensor *bias) {
    x = ggml_add(ctx, x, bias);
    return ggml_leaky_relu(ctx, x, 0.2f, false);
}

// conv2d with per-input-channel modulation folded into the GEMM weight
// matrix (avoids a full-map x*style pass): rows of A scale by style.
// a [KW,KH,IC,OC], style [IC,1] f32, b [W,H,IC,N]
inline ggml_tensor *conv2d_mod(ggml_context *ctx, ggml_tensor *a, ggml_tensor *style,
                               ggml_tensor *b, int s, int p) {
    const int64_t KK = a->ne[0] * a->ne[1];
    ggml_tensor *sty = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, style, 1, a->ne[2]),
                                      KK, a->ne[2], 1, 1);
    sty = ggml_reshape_2d(ctx, sty, KK * a->ne[2], 1);
    ggml_tensor *am = ggml_reshape_2d(ctx, a, KK * a->ne[2], a->ne[3]);
    if (a->type != GGML_TYPE_F32) am = ggml_cast(ctx, am, GGML_TYPE_F32);
    am = ggml_mul(ctx, am, sty);
    if (a->type != GGML_TYPE_F32) am = ggml_cast(ctx, am, a->type);
    ggml_tensor *im = ggml_im2col(ctx, a, b, s, s, p, p, 1, 1, true, a->type);
    ggml_tensor *r = ggml_mul_mat(
        ctx, am, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]));
    r = ggml_reshape_3d(ctx, r, a->ne[3], im->ne[1], im->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, r, 2, 0, 1, 3));  // [OW, OH, OC]
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
