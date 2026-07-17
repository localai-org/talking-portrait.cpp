#include "motion_dec.h"

#include "ops_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tp {

// stage channels: convs[2j] IC->OC upsample, convs[2j+1] OC->OC
static const int ST_IC[7] = {512, 512, 512, 512, 256, 128, 64};
static const int ST_OC[7] = {512, 512, 512, 256, 128, 64, 32};

// tensors whose (scale-folded) f32 data must stay on the host at load so
// prepare() can build flipped/demod-reduction derived tensors
std::vector<std::string> motion_dec_host_tensors() {
    std::vector<std::string> v = {"dec.conv1.conv.weight"};
    for (int i = 0; i < 14; i++) {
        char nm[48];
        snprintf(nm, sizeof nm, "dec.convs.%d.conv.weight", i);
        v.push_back(nm);
        if (i % 2 == 0) {
            snprintf(nm, sizeof nm, "dec.convs.%d.conv.blur.kernel", i);
            v.push_back(nm);
        }
    }
    return v;
}

bool motion_dec::prepare(backend_t &be, std::string &err) {
    // derived: S_aug demod matrices for all 15 demod convs, spatially
    // flipped weights for the 7 upsample convs
    ggml_init_params cp = {
        /*mem_size  =*/ ggml_tensor_overhead() * 64,
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    dctx = ggml_init(cp);

    struct job {
        std::string name;
        int ic, oc;
        bool flip;
    };
    std::vector<job> jobs;
    jobs.push_back({"dec.conv1.conv.weight", 512, 512, false});
    for (int j = 0; j < 7; j++) {
        char a[48], b[48];
        snprintf(a, sizeof a, "dec.convs.%d.conv.weight", 2 * j);
        snprintf(b, sizeof b, "dec.convs.%d.conv.weight", 2 * j + 1);
        jobs.push_back({a, ST_IC[j], ST_OC[j], true});
        jobs.push_back({b, ST_OC[j], ST_OC[j], false});
    }

    for (auto &jb : jobs) {
        auto it = w->host.find(jb.name);
        if (it == w->host.end()) {
            err = "host copy missing for " + jb.name;
            return false;
        }
        ggml_tensor *sa = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, jb.ic + 1, jb.oc);
        ggml_set_name(sa, ("saug." + jb.name).c_str());
        derived["saug." + jb.name] = sa;
        const ggml_type wt = w->keep_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (jb.flip) {
            // sub-pixel composed kernel, cf-ordered rows r = ic + IC*(kw+3kh),
            // 4 output-parity classes stacked on OC
            ggml_tensor *ts = ggml_new_tensor_2d(dctx, wt, (int64_t) jb.ic * 9, 4 * jb.oc);
            ggml_set_name(ts, ("tsubcf." + jb.name).c_str());
            derived["tsubcf." + jb.name] = ts;
            // spatially flipped weights for the numerically-conservative
            // parity builder (conv_transpose as conv on zero-stuffed input)
            ggml_tensor *wf = ggml_new_tensor_4d(dctx, GGML_TYPE_F32, 3, 3, jb.ic, jb.oc);
            ggml_set_name(wf, ("wflip." + jb.name).c_str());
            derived["wflip." + jb.name] = wf;
        } else if (jb.name != "dec.conv1.conv.weight") {
            // cf-ordered GEMM matrix for the dense 3x3 convs
            ggml_tensor *ac = ggml_new_tensor_2d(dctx, wt, (int64_t) jb.ic * 9, jb.oc);
            ggml_set_name(ac, ("acf." + jb.name).c_str());
            derived["acf." + jb.name] = ac;
        }
    }
    dbuf = ggml_backend_alloc_ctx_tensors(dctx, be.be);
    if (!dbuf) {
        err = "failed to allocate derived buffer";
        return false;
    }

    std::vector<float> buf;
    for (auto &jb : jobs) {
        const std::vector<float> &W = w->host[jb.name];  // [OC,IC,3,3] torch order
        const int IC = jb.ic, OC = jb.oc;
        if ((int64_t) W.size() != (int64_t) OC * IC * 9) {
            err = jb.name + ": unexpected size";
            return false;
        }
        // S and eps are HALVED: divides the demod denominator by sqrt(2),
        // folding the FusedLeakyReLU sqrt(2) into the demodulation
        // (activate.bias gets the matching sqrt(2) at load)
        buf.assign((size_t) (IC + 1) * OC, 0.0f);
        for (int o = 0; o < OC; o++) {
            for (int i = 0; i < IC; i++) {
                double s = 0;
                const float *pw = W.data() + ((size_t) o * IC + i) * 9;
                for (int k = 0; k < 9; k++) s += (double) pw[k] * pw[k];
                buf[(size_t) o * (IC + 1) + i] = (float) (s * 0.5);
            }
            buf[(size_t) o * (IC + 1) + IC] = 0.5e-8f;
        }
        ggml_tensor *sa = derived["saug." + jb.name];
        ggml_backend_tensor_set(sa, buf.data(), 0, ggml_nbytes(sa));
        auto upload = [&](ggml_tensor *t, const std::vector<float> &v) {
            if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> h(v.size());
                for (size_t i = 0; i < v.size(); i++) h[i] = ggml_fp32_to_fp16(v[i]);
                ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
            } else {
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            }
        };
        if (!jb.flip && jb.name != "dec.conv1.conv.weight") {
            // A_cf[r + R*oc] = W[oc][ic][kh][kw], r = ic + IC*(kw + 3*kh)
            const int R = IC * 9;
            std::vector<float> ac((size_t) R * OC);
            for (int o = 0; o < OC; o++)
                for (int i = 0; i < IC; i++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                            ac[(size_t) o * R + i + (size_t) IC * (kw + 3 * kh)] =
                                W[((size_t) o * IC + i) * 9 + kh * 3 + kw];
            upload(derived["acf." + jb.name], ac);
        }
        if (jb.flip) {
            // conv_transpose(3x3,s2) followed by Blur(4x4 FIR) composes into a
            // 6x6/stride-2 kernel; per output parity class (by,bx) that is an
            // ordinary 3x3 pad-1 conv of the un-upsampled input:
            //   out[2m+by, 2n+bx] = conv3x3(x, T_class)[m, n]
            //   T[dy,dx] = sum_{cy,cx} V[cy,cx] * Bk[3-by-2dy-cy, 3-bx-2dx-cx]
            // with V = spatially flipped modconv weight, d = 1 - kernel index.
            const std::string blur_nm =
                jb.name.substr(0, jb.name.size() - strlen("weight")) + "blur.kernel";
            auto bit = w->host.find(blur_nm);
            if (bit == w->host.end() || bit->second.size() != 16) {
                err = "host copy missing for " + blur_nm;
                return false;
            }
            const std::vector<float> &BK = bit->second;  // [4,4], includes x4
            const int R = IC * 9;
            std::vector<float> tb((size_t) R * 4 * OC, 0.0f);
            for (int o = 0; o < OC; o++) {
                for (int i = 0; i < IC; i++) {
                    const float *pw = W.data() + ((size_t) o * IC + i) * 9;
                    float V[3][3];
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++) V[a][b] = pw[(2 - a) * 3 + (2 - b)];
                    for (int by = 0; by < 2; by++)
                        for (int bx = 0; bx < 2; bx++) {
                            const int k = by * 2 + bx;
                            // column = k*OC + o; row = ic + IC*(kw + 3*kh)
                            float *dst = tb.data() + (size_t) (k * OC + o) * R;
                            for (int ty = 0; ty < 3; ty++)
                                for (int tx = 0; tx < 3; tx++) {
                                    const int dy = 1 - ty, dx = 1 - tx;
                                    double acc = 0;
                                    for (int cy = 0; cy < 3; cy++) {
                                        const int ay = 3 - by - 2 * dy - cy;
                                        if (ay < 0 || ay > 3) continue;
                                        for (int cx = 0; cx < 3; cx++) {
                                            const int ax = 3 - bx - 2 * dx - cx;
                                            if (ax < 0 || ax > 3) continue;
                                            acc += (double) V[cy][cx] *
                                                   (double) BK[(size_t) ay * 4 + ax];
                                        }
                                    }
                                    dst[i + (size_t) IC * (tx + 3 * ty)] = (float) acc;
                                }
                        }
                }
            }
            upload(derived["tsubcf." + jb.name], tb);
            std::vector<float> fb(W.size());
            for (int o = 0; o < OC; o++)
                for (int i = 0; i < IC; i++)
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            fb[(((size_t) o * IC + i) * 3 + a) * 3 + b] =
                                W[(((size_t) o * IC + i) * 3 + (2 - a)) * 3 + (2 - b)];
            ggml_tensor *wf = derived["wflip." + jb.name];
            ggml_backend_tensor_set(wf, fb.data(), 0, ggml_nbytes(wf));
        }
    }
    return true;
}

// per-frame graph build ------------------------------------------------

struct mdec_build {
    plan_t &p;
    ggml_context *ctx;
    weights_t *w;
    std::map<std::string, ggml_tensor *> &derived;
    ggml_tensor *latent = nullptr;  // [512,1]
    ggml_tensor *one = nullptr;     // [1,1] const 1.0
    std::string err;

    // Dataflow is CHANNEL-LAST ([C, N] GEMM outputs); spatial [W,H,C] is
    // materialized only where im2col / FIR / taps need it.

    ggml_tensor *need(const std::string &n) {
        ggml_tensor *t = w->get(n);
        if (!t && err.empty()) err = "missing tensor " + n;
        return t;
    }
    ggml_tensor *der(const std::string &n) {
        auto it = derived.find(n);
        if (it == derived.end()) {
            if (err.empty()) err = "missing derived tensor " + n;
            return nullptr;
        }
        return it->second;
    }
    ggml_tensor *style_of(const std::string &base) {
        ggml_tensor *mw = need(base + ".modulation.weight");
        ggml_tensor *mb = need(base + ".modulation.bias");
        if (!mw || !mb) return nullptr;
        return linear(ctx, mw, mb, latent);  // [IC,1]
    }
    // demodulation denominator sqrt(S s^2 + eps) [OC,1]; S is pre-halved so
    // this also folds the FusedLeakyReLU sqrt(2)
    ggml_tensor *demod_den(const std::string &wname, ggml_tensor *style) {
        ggml_tensor *sa = der("saug." + wname);
        if (!sa) return nullptr;
        ggml_tensor *s2 = ggml_concat(ctx, ggml_sqr(ctx, style), one, 0);  // [IC+1,1]
        return ggml_sqrt(ctx, ggml_mul_mat(ctx, sa, s2));                  // [OC,1]
    }
    // GEMM A-matrix with optional per-in-channel style and per-out-channel
    // 1/den folded in (weight-sized passes instead of full-map passes)
    ggml_tensor *gemm_weight(ggml_tensor *W4d, ggml_tensor *style, ggml_tensor *den) {
        const int64_t KK = W4d->ne[0] * W4d->ne[1], IC = W4d->ne[2], OC = W4d->ne[3];
        ggml_tensor *A = ggml_reshape_2d(ctx, W4d, KK * IC, OC);
        if (style || den) {
            if (A->type != GGML_TYPE_F32) A = ggml_cast(ctx, A, GGML_TYPE_F32);
            if (style) {
                ggml_tensor *sty = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, style, 1, IC),
                                                  KK, IC, 1, 1);
                A = ggml_mul(ctx, A, ggml_reshape_2d(ctx, sty, KK * IC, 1));
            }
            if (den) A = ggml_div(ctx, A, ggml_reshape_2d(ctx, den, 1, OC));
            if (W4d->type != GGML_TYPE_F32) A = ggml_cast(ctx, A, W4d->type);
        }
        return A;
    }
    // conv (stride 1) producing channel-last [OC, N]
    ggml_tensor *conv_cl(ggml_tensor *W4d, ggml_tensor *style, ggml_tensor *den,
                         ggml_tensor *x_sp, int pad) {
        ggml_tensor *A = gemm_weight(W4d, style, den);
        ggml_tensor *im = ggml_im2col(ctx, W4d, x_sp, 1, 1, pad, pad, 1, 1, true, W4d->type);
        return ggml_mul_mat(
            ctx, A, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]));
    }
    // channel-last [C, N] -> spatial [W, H, C]
    ggml_tensor *cl_to_sp(ggml_tensor *t_cl, int64_t Wd, int64_t Hd) {
        ggml_tensor *t = ggml_reshape_3d(ctx, t_cl, t_cl->ne[0], Wd, Hd);
        return ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));
    }
    // FusedLeakyReLU on channel-last (sqrt2 folded upstream); ab [1,1,C,1]
    ggml_tensor *bias_lrelu_cl(ggml_tensor *y_cl, ggml_tensor *ab) {
        y_cl = ggml_add(ctx, y_cl, ggml_reshape_2d(ctx, ab, ab->ne[2], 1));
        return ggml_leaky_relu(ctx, y_cl, 0.2f, false);
    }
    // weight-side folding pays off only when the map is larger than the
    // weight matrix (big resolutions); small stages modulate the map
    static bool fold_into_weights(int64_t wsz, int64_t N, int64_t ic, int64_t oc) {
        return 4 * wsz < 2 * N * (ic + oc);
    }
    // fold style (per-in-channel, cf tiling) and 1/den (per-out-channel)
    // into a cf-ordered GEMM matrix A [R, OC']
    ggml_tensor *fold_A(ggml_tensor *A, ggml_tensor *style, ggml_tensor *den, int64_t IC) {
        if (!style && !den) return A;
        const ggml_type t0 = A->type;
        ggml_tensor *Af = t0 == GGML_TYPE_F32 ? A : ggml_cast(ctx, A, GGML_TYPE_F32);
        if (style) {
            const int64_t KK = A->ne[0] / IC;
            ggml_tensor *sty = ggml_repeat_4d(ctx, style, IC * KK, 1, 1, 1);  // tile
            Af = ggml_mul(ctx, Af, sty);
        }
        if (den) Af = ggml_div(ctx, Af, ggml_reshape_2d(ctx, den, 1, den->ne[0]));
        return t0 == GGML_TYPE_F32 ? Af : ggml_cast(ctx, Af, t0);
    }
    // channels-first conv: x_cl [IC, N] over a WxH map, A [IC*9, OC'] -> [OC', N]
    ggml_tensor *conv_cf(ggml_tensor *A, ggml_tensor *x_cl, int64_t W, int64_t H, int pad) {
        const int64_t IC = x_cl->ne[0];
        ggml_tensor *x3 = ggml_reshape_3d(ctx, x_cl, IC, W, H);
        ggml_tensor *shape4 = ggml_reshape_4d(ctx, A, 3, 3, IC, A->ne[1]);  // dims only
        ggml_tensor *im = ggml_im2col_cf(ctx, shape4, x3, 1, 1, pad, pad, 1, 1, A->type);
        return ggml_mul_mat(ctx, A,
                            ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]));
    }
    // normal modulated conv 3x3 (demodulate) + FLReLU: CL in, CL out
    ggml_tensor *styled_conv(ggml_tensor *x_cl, int64_t W, int64_t H, const std::string &base) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *A = der("acf." + base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !A || !ab) return nullptr;
        ggml_tensor *den = demod_den(base + ".conv.weight", style);
        if (!den) return nullptr;
        const int64_t N = W * H, IC = x_cl->ne[0];
        ggml_tensor *y;
        if (fold_into_weights(A->ne[0] * A->ne[1], N, IC, A->ne[1])) {
            y = conv_cf(fold_A(A, style, den, IC), x_cl, W, H, 1);
        } else {
            y = conv_cf(A, ggml_mul(ctx, x_cl, style), W, H, 1);
            y = ggml_div(ctx, y, den);
        }
        return bias_lrelu_cl(y, ab);
    }
    // first 4x4 styled conv (spatial const input, tiny)
    ggml_tensor *styled_conv_input(ggml_tensor *x_sp, const std::string &base) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *W = need(base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !W || !ab) return nullptr;
        ggml_tensor *den = demod_den(base + ".conv.weight", style);
        if (!den) return nullptr;
        ggml_tensor *xs = ggml_mul(ctx, x_sp, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        ggml_tensor *y = conv_cl(W, nullptr, nullptr, xs, 1);  // [OC, N]
        y = ggml_div(ctx, y, den);
        return bias_lrelu_cl(y, ab);
    }
    // upsampling modulated conv: sub-pixel form of conv_transpose(s2)+Blur —
    // one 3x3 pad-1 conv, 4 parity classes stacked on OC, interleaved in
    // channel-last (ne3 kept 1: CUDA concat launches per ne3 slice)
    ggml_tensor *styled_conv_up(ggml_tensor *x_cl, int64_t W, int64_t H, const std::string &base) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *Ts = der("tsubcf." + base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !Ts || !ab) return nullptr;
        ggml_tensor *den = demod_den(base + ".conv.weight", style);
        if (!den) return nullptr;
        const int64_t OC = Ts->ne[1] / 4, IC = x_cl->ne[0];
        const int64_t N = W * H;
        ggml_tensor *y4;
        if (fold_into_weights(Ts->ne[0] * Ts->ne[1], N, IC, Ts->ne[1])) {
            ggml_tensor *den4 = ggml_repeat_4d(ctx, den, 4 * OC, 1, 1, 1);
            y4 = conv_cf(fold_A(Ts, style, den4, IC), x_cl, W, H, 1);
        } else {
            y4 = conv_cf(Ts, ggml_mul(ctx, x_cl, style), W, H, 1);
            y4 = ggml_div(ctx, y4, ggml_repeat_4d(ctx, den, 4 * OC, 1, 1, 1));
        }
        ggml_tensor *cls[4];
        for (int k = 0; k < 4; k++)
            cls[k] = ggml_cont(ctx, ggml_view_2d(ctx, y4, OC, N, y4->nb[1],
                                                 (size_t) k * OC * 4));
        // u-interleave: (o, bx, m) -> flat o + OC*bx + 2OC*m
        ggml_tensor *E[2];
        for (int by = 0; by < 2; by++) {
            ggml_tensor *a = ggml_reshape_3d(ctx, cls[by * 2 + 0], OC, 1, N);
            ggml_tensor *b = ggml_reshape_3d(ctx, cls[by * 2 + 1], OC, 1, N);
            // [OC,2,N] == flat o + OC*bx + 2OC*(x + W*y) -> [(o,u), y]
            E[by] = ggml_reshape_2d(ctx, ggml_concat(ctx, a, b, 1), OC * 2 * W, H);
        }
        // v-interleave: (q, by, y) -> flat q + 2OCW*by + 4OCW*y
        ggml_tensor *F = ggml_concat(ctx, ggml_reshape_3d(ctx, E[0], OC * 2 * W, 1, H),
                                     ggml_reshape_3d(ctx, E[1], OC * 2 * W, 1, H), 1);
        F = ggml_reshape_2d(ctx, F, OC, 4 * N);  // CL over the 2W x 2H map
        return bias_lrelu_cl(F, ab);
    }
    // upfirdn2d up=2 pad(2,1) with a x4 kernel (Upsample module, 3-channel)
    ggml_tensor *upsample2x(ggml_tensor *x, ggml_tensor *kern, ggml_tensor *mask2x) {
        const int C = (int) x->ne[2];
        ggml_tensor *zs = zero_stuff2(ctx, x, mask2x);
        zs = ggml_pad_ext(ctx, zs, 2, 1, 2, 1, 0, 0, 0, 0);
        ggml_tensor *k4 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, kern, 4, 4, 1, 1), 4, 4, 1, C);
        return ggml_conv_2d_dw_direct(ctx, k4, zs, 1, 1, 0, 0, 1, 1);
    }
    // ---- numerically-conservative (parity) variants: spatial layout,
    // direct convolution kernels, map-side modulation ----
    ggml_tensor *styled_conv_exact(ggml_tensor *x_sp, const std::string &base) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *W = need(base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !W || !ab) return nullptr;
        ggml_tensor *den = demod_den(base + ".conv.weight", style);
        if (!den) return nullptr;
        ggml_tensor *xs = ggml_mul(ctx, x_sp, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        ggml_tensor *y = ggml_conv_2d_direct(ctx, W, xs, 1, 1, 1, 1, 1, 1);
        y = ggml_div(ctx, y, ggml_reshape_4d(ctx, den, 1, 1, den->ne[0], 1));
        return fused_lrelu(ctx, y, ab);
    }
    ggml_tensor *styled_conv_up_exact(ggml_tensor *x_sp, const std::string &base,
                                      ggml_tensor *mask2x) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *Wf = der("wflip." + base + ".conv.weight");
        ggml_tensor *kb = need(base + ".conv.blur.kernel");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !Wf || !kb || !ab) return nullptr;
        ggml_tensor *den = demod_den(base + ".conv.weight", style);
        if (!den) return nullptr;
        const int OC = (int) Wf->ne[3];
        ggml_tensor *xs = ggml_mul(ctx, x_sp, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        ggml_tensor *zs = zero_stuff2(ctx, xs, mask2x);
        zs = ggml_pad_ext(ctx, zs, 2, 1, 2, 1, 0, 0, 0, 0);
        ggml_tensor *y = ggml_conv_2d_direct(ctx, Wf, zs, 1, 1, 0, 0, 1, 1);
        y = ggml_div(ctx, y, ggml_reshape_4d(ctx, den, 1, 1, den->ne[0], 1));
        ggml_tensor *k4 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, kb, 4, 4, 1, 1), 4, 4, 1, OC);
        y = blur(ctx, y, k4, 1, 1);
        return fused_lrelu(ctx, y, ab);
    }
    // 1x1 modulated conv on a spatial map (direct kernel)
    ggml_tensor *mod_1x1_exact(ggml_tensor *W1, ggml_tensor *style, ggml_tensor *x_sp) {
        ggml_tensor *xs = ggml_mul(ctx, x_sp, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        return ggml_conv_2d_direct(ctx, W1, xs, 1, 1, 0, 0, 1, 1);
    }
    // spatial [W,H,C] -> channel-last [C, N]
    ggml_tensor *sp_to_cl(ggml_tensor *t_sp) {
        ggml_tensor *t = ggml_cont(ctx, ggml_permute(ctx, t_sp, 1, 2, 0, 3));
        return ggml_reshape_2d(ctx, t, t->ne[0], t->ne[1] * t->ne[2]);
    }

    // bilinear zeros-padding grid_sample of feat_t [C, r*r]: returns CL [C, r*r]
    ggml_tensor *grid_sample_cl(ggml_tensor *feat_t, ggml_tensor *gx, ggml_tensor *gy, int r) {
        const int64_t NN = (int64_t) r * r;
        ggml_tensor *grid = ggml_concat(ctx, ggml_reshape_2d(ctx, gx, 1, NN),
                                        ggml_reshape_2d(ctx, gy, 1, NN), 0);  // [2, N]
        return ggml_bilinear_gather(ctx, feat_t, grid, r, r);
    }
};

bool motion_dec::build(backend_t &be, bool with_taps, std::string &err) {
    frame.release();

    // persistent constants: allocated once in their own backend buffer so
    // gallocr can never hand their memory to later graph nodes
    ggml_tensor *masks[7], *xs_x[7], *xs_y[7], *feats[7], *onec;
    if (!cctx) {
        ggml_init_params cp = {ggml_tensor_overhead() * 32, nullptr, true};
        cctx = ggml_init(cp);
        auto mk = [&](const char *nm, int64_t a, int64_t b,
                      ggml_type tt = GGML_TYPE_F32) {
            ggml_tensor *t = ggml_new_tensor_2d(cctx, tt, a, b);
            ggml_set_name(t, nm);
            consts[nm] = t;
            return t;
        };
        for (int j = 0; j < 7; j++) {
            const int r = 8 << j;
            char nm[24];
            snprintf(nm, sizeof nm, "mask%d", j);
            mk(nm, r, r);
            snprintf(nm, sizeof nm, "xsx%d", j);
            mk(nm, r, r);
            snprintf(nm, sizeof nm, "xsy%d", j);
            mk(nm, r, r);
            snprintf(nm, sizeof nm, "feat%d", j);
            // channels of feat j == ST_OC[j] (conv{6-j} channels match stage OC)
            mk(nm, ST_OC[j], (int64_t) r * r, w->keep_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        }
        mk("one", 1, 1);
        cbuf = ggml_backend_alloc_ctx_tensors(cctx, be.be);
        if (!cbuf) {
            err = "failed to allocate mdec const buffer";
            return false;
        }
        // fill masks / identity grids / one
        std::vector<float> buf;
        const float one = 1.0f;
        ggml_backend_tensor_set(consts["one"], &one, 0, 4);
        for (int j = 0; j < 7; j++) {
            const int r = 8 << j;
            char nm[24];
            buf.assign((size_t) r * r, 0.0f);
            for (int y = 0; y < r; y += 2)
                for (int x = 0; x < r; x += 2) buf[(size_t) y * r + x] = 1.0f;
            snprintf(nm, sizeof nm, "mask%d", j);
            ggml_backend_tensor_set(consts[nm], buf.data(), 0, (size_t) r * r * 4);
            std::vector<double> lin((size_t) r);
            for (int i = 0; i < r; i++) lin[(size_t) i] = -1.0 + 2.0 * i / (r - 1);
            for (int y = 0; y < r; y++)
                for (int x = 0; x < r; x++) buf[(size_t) y * r + x] = (float) lin[(size_t) x];
            snprintf(nm, sizeof nm, "xsx%d", j);
            ggml_backend_tensor_set(consts[nm], buf.data(), 0, (size_t) r * r * 4);
            for (int y = 0; y < r; y++)
                for (int x = 0; x < r; x++) buf[(size_t) y * r + x] = (float) lin[(size_t) y];
            snprintf(nm, sizeof nm, "xsy%d", j);
            ggml_backend_tensor_set(consts[nm], buf.data(), 0, (size_t) r * r * 4);
        }
    }
    for (int j = 0; j < 7; j++) {
        char nm[24];
        snprintf(nm, sizeof nm, "mask%d", j);
        masks[j] = consts[nm];
        snprintf(nm, sizeof nm, "xsx%d", j);
        xs_x[j] = consts[nm];
        snprintf(nm, sizeof nm, "xsy%d", j);
        xs_y[j] = consts[nm];
        snprintf(nm, sizeof nm, "feat%d", j);
        feats[j] = consts[nm];
    }
    onec = consts["one"];

    plan_t &p = frame;
    p.begin(4096);
    ggml_context *ctx = p.ctx;
    mdec_build B{p, ctx, w, derived};

    B.latent = p.input("style", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1));
    B.one = onec;

    std::vector<tap_entry> taps;
    auto reg = [&](const std::string &name, ggml_tensor *t, std::vector<int64_t> sh) {
        if (!with_taps) return;
        char on[48];
        snprintf(on, sizeof on, "tapout.%zu", taps.size());
        p.output(on, t);
        taps.push_back({name, t, std::move(sh)});
    };

    ggml_tensor *skip = nullptr;       // rgb accumulator (spatial, 3ch)
    ggml_tensor *skip_flow = nullptr;  // flowmap accumulator (spatial, 3ch)

    // debug knob: truncate the pyramid for per-stage cost bisection
    int max_stage = 7;
    if (const char *e = getenv("TP_MDEC_STAGES")) {
        max_stage = atoi(e);
        if (max_stage < 1 || max_stage > 7) max_stage = 7;
    }

    // TP_MDEC_EXACT=1 forces the conservative graph for generation too:
    // measurement knob for backends where the fast graph's custom ops fall
    // back to the CPU (e.g. Vulkan). Taps stay tied to with_taps.
    const bool exact_graph = with_taps || getenv("TP_MDEC_EXACT") != nullptr;

    if (exact_graph) {
        // ---- numerically-conservative parity graph (spatial, direct convs,
        // map-side modulation): per-frame cost does not matter here, tap
        // fidelity does. The fast channel-last graph below is used for
        // generation and is gated end-to-end by M6 SSIM. ----
        ggml_tensor *cinp = B.need("dec.input.input");
        ggml_tensor *out = cinp ? B.styled_conv_exact(cinp, "dec.conv1") : nullptr;
        if (!out) {
            err = B.err.empty() ? "exact conv1 failed" : B.err;
            return false;
        }
        reg("conv1_4", out, {1, 512, 4, 4});
        for (int j = 0; j < max_stage; j++) {
            const int r = 8 << j;
            char b1[32], b2[32], bf[32], br[32], tn[32];
            snprintf(b1, sizeof b1, "dec.convs.%d", 2 * j);
            snprintf(b2, sizeof b2, "dec.convs.%d", 2 * j + 1);
            snprintf(bf, sizeof bf, "dec.to_flows.%d", j);
            snprintf(br, sizeof br, "dec.to_rgbs.%d", j);

            out = B.styled_conv_up_exact(out, b1, masks[j]);
            if (!out) {
                err = B.err;
                return false;
            }
            snprintf(tn, sizeof tn, "r%d.conv1", r);
            reg(tn, out, {1, ST_OC[j], r, r});
            out = B.styled_conv_exact(out, b2);
            if (!out) {
                err = B.err;
                return false;
            }
            snprintf(tn, sizeof tn, "r%d.conv2", r);
            reg(tn, out, {1, ST_OC[j], r, r});

            const std::string fbase = bf;
            ggml_tensor *fstyle = B.style_of(fbase + ".conv");
            ggml_tensor *fW = B.need(fbase + ".conv.weight");
            ggml_tensor *fbias = B.need(fbase + ".bias");
            ggml_tensor *fup = B.need(fbase + ".upsample.kernel");
            if (!fstyle || !fW || !fbias || !fup) {
                err = B.err;
                return false;
            }
            ggml_tensor *fl = B.mod_1x1_exact(fW, fstyle, out);
            fl = ggml_add(ctx, fl, fbias);
            if (skip_flow) fl = ggml_add(ctx, fl, B.upsample2x(skip_flow, fup, masks[j]));
            skip_flow = fl;
            snprintf(tn, sizeof tn, "r%d.flowmap", r);
            reg(tn, fl, {1, 3, r, r});

            ggml_tensor *ch0 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], 0));
            ggml_tensor *ch1 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], fl->nb[2]));
            ggml_tensor *ch2 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], 2 * fl->nb[2]));
            ggml_tensor *gx = ggml_add(ctx, ggml_tanh(ctx, ch0), xs_x[j]);
            ggml_tensor *gy = ggml_add(ctx, ggml_tanh(ctx, ch1), xs_y[j]);
            {
                ggml_tensor *grid = ggml_concat(ctx, ggml_reshape_3d(ctx, gx, 1, r, r),
                                                ggml_reshape_3d(ctx, gy, 1, r, r), 0);
                snprintf(tn, sizeof tn, "r%d.grid", r);
                reg(tn, grid, {1, r, r, 2});
            }
            ggml_tensor *m = ggml_sigmoid(ctx, ch2);
            ggml_tensor *sampled = B.cl_to_sp(B.grid_sample_cl(feats[j], gx, gy, r), r, r);
            ggml_tensor *m3 = ggml_reshape_3d(ctx, m, r, r, 1);
            ggml_tensor *warp = ggml_mul(ctx, sampled, m3);
            snprintf(tn, sizeof tn, "r%d.warp", r);
            reg(tn, warp, {1, ST_OC[j], r, r});
            ggml_tensor *one_m =
                ggml_reshape_3d(ctx, ggml_scale_bias(ctx, m, -1.0f, 1.0f), r, r, 1);
            ggml_tensor *trunk = ggml_add(ctx, warp, ggml_mul(ctx, out, one_m));
            snprintf(tn, sizeof tn, "r%d.out", r);
            reg(tn, trunk, {1, ST_OC[j], r, r});

            const std::string rbase = br;
            ggml_tensor *rW = B.need(rbase + ".conv.0.weight");
            ggml_tensor *rlb = B.need(rbase + ".conv.1.bias");
            ggml_tensor *rbias = B.need(rbase + ".bias");
            ggml_tensor *rup = B.need(rbase + ".upsample.kernel");
            if (!rW || !rlb || !rbias || !rup) {
                err = B.err;
                return false;
            }
            ggml_tensor *rgb = ggml_conv_2d_direct(ctx, rW, warp, 1, 1, 0, 0, 1, 1);
            rgb = fused_lrelu(ctx, rgb, rlb);
            rgb = ggml_add(ctx, rgb, rbias);
            if (skip) rgb = ggml_add(ctx, rgb, B.upsample2x(skip, rup, masks[j]));
            skip = rgb;
            snprintf(tn, sizeof tn, "r%d.rgb", r);
            reg(tn, rgb, {1, 3, r, r});

            out = trunk;
        }
    } else {
        // constant input + conv1 + (no to_rgb1 in forward)
        ggml_tensor *cinp = B.need("dec.input.input");  // [4,4,512,1]
        if (!cinp) {
            err = B.err;
            return false;
        }
        ggml_tensor *x_cl = B.styled_conv_input(cinp, "dec.conv1");  // [512, 16]
        if (!x_cl) {
            err = B.err;
            return false;
        }
        if (with_taps) reg("conv1_4", B.cl_to_sp(x_cl, 4, 4), {1, 512, 4, 4});

        for (int j = 0; j < max_stage; j++) {
            const int r = 8 << j;
            const int64_t N = (int64_t) r * r;
            char b1[32], b2[32], bf[32], br[32];
            snprintf(b1, sizeof b1, "dec.convs.%d", 2 * j);
            snprintf(b2, sizeof b2, "dec.convs.%d", 2 * j + 1);
            snprintf(bf, sizeof bf, "dec.to_flows.%d", j);
            snprintf(br, sizeof br, "dec.to_rgbs.%d", j);
            char tn[32];

            ggml_tensor *c1_cl = B.styled_conv_up(x_cl, r / 2, r / 2, b1);  // [OC, N]
            if (!c1_cl) {
                err = B.err;
                return false;
            }
            if (with_taps) {
                snprintf(tn, sizeof tn, "r%d.conv1", r);
                reg(tn, B.cl_to_sp(c1_cl, r, r), {1, ST_OC[j], r, r});
            }
            ggml_tensor *out_cl = B.styled_conv(c1_cl, r, r, b2);  // [OC, N]
            if (!out_cl) {
                err = B.err;
                return false;
            }
            if (with_taps) {
                snprintf(tn, sizeof tn, "r%d.conv2", r);
                reg(tn, B.cl_to_sp(out_cl, r, r), {1, ST_OC[j], r, r});
            }

            // ---- ToFlow: 1x1 modulated conv == plain GEMM on the CL tensor ----
            const std::string fbase = bf;
            ggml_tensor *fstyle = B.style_of(fbase + ".conv");
            ggml_tensor *fW = B.need(fbase + ".conv.weight");
            ggml_tensor *fbias = B.need(fbase + ".bias");
            ggml_tensor *fup = B.need(fbase + ".upsample.kernel");
            if (!fstyle || !fW || !fbias || !fup) {
                err = B.err;
                return false;
            }
            ggml_tensor *fl_cl = ggml_mul_mat(ctx, B.gemm_weight(fW, fstyle, nullptr), out_cl);
            fl_cl = ggml_add(ctx, fl_cl, ggml_reshape_2d(ctx, fbias, 3, 1));
            ggml_tensor *fl = B.cl_to_sp(fl_cl, r, r);  // [r,r,3]
            if (skip_flow) fl = ggml_add(ctx, fl, B.upsample2x(skip_flow, fup, masks[j]));
            skip_flow = fl;
            snprintf(tn, sizeof tn, "r%d.flowmap", r);
            reg(tn, fl, {1, 3, r, r});

            ggml_tensor *ch0 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], 0));
            ggml_tensor *ch1 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], fl->nb[2]));
            ggml_tensor *ch2 = ggml_cont(ctx, ggml_view_2d(ctx, fl, r, r, fl->nb[1], 2 * fl->nb[2]));
            ggml_tensor *gx = ggml_add(ctx, ggml_tanh(ctx, ch0), xs_x[j]);
            ggml_tensor *gy = ggml_add(ctx, ggml_tanh(ctx, ch1), xs_y[j]);
            if (with_taps) {
                ggml_tensor *grid = ggml_concat(ctx, ggml_reshape_3d(ctx, gx, 1, r, r),
                                                ggml_reshape_3d(ctx, gy, 1, r, r), 0);
                snprintf(tn, sizeof tn, "r%d.grid", r);
                reg(tn, grid, {1, r, r, 2});
            }
            ggml_tensor *m = ggml_sigmoid(ctx, ch2);  // [r,r]
            ggml_tensor *m_row = ggml_reshape_2d(ctx, m, 1, N);
            ggml_tensor *sampled = B.grid_sample_cl(feats[j], gx, gy, r);  // [C, N]
            ggml_tensor *warp_cl = ggml_mul(ctx, sampled, m_row);
            if (with_taps) {
                snprintf(tn, sizeof tn, "r%d.warp", r);
                reg(tn, B.cl_to_sp(warp_cl, r, r), {1, ST_OC[j], r, r});
            }
            ggml_tensor *one_m_row =
                ggml_reshape_2d(ctx, ggml_scale_bias(ctx, m, -1.0f, 1.0f), 1, N);
            ggml_tensor *trunk_cl = ggml_add(ctx, warp_cl, ggml_mul(ctx, out_cl, one_m_row));
            if (with_taps) {
                snprintf(tn, sizeof tn, "r%d.out", r);
                reg(tn, B.cl_to_sp(trunk_cl, r, r), {1, ST_OC[j], r, r});
            }

            // ---- ToRGB(out_warp, skip): 1x1 EqualConv2d == GEMM on warp_cl ----
            const std::string rbase = br;
            ggml_tensor *rW = B.need(rbase + ".conv.0.weight");
            ggml_tensor *rlb = B.need(rbase + ".conv.1.bias");
            ggml_tensor *rbias = B.need(rbase + ".bias");
            ggml_tensor *rup = B.need(rbase + ".upsample.kernel");
            if (!rW || !rlb || !rbias || !rup) {
                err = B.err;
                return false;
            }
            ggml_tensor *rgb_cl = ggml_mul_mat(ctx, B.gemm_weight(rW, nullptr, nullptr), warp_cl);
            rgb_cl = B.bias_lrelu_cl(rgb_cl, rlb);
            rgb_cl = ggml_add(ctx, rgb_cl, ggml_reshape_2d(ctx, rbias, 3, 1));
            ggml_tensor *rgb = B.cl_to_sp(rgb_cl, r, r);
            if (skip) rgb = ggml_add(ctx, rgb, B.upsample2x(skip, rup, masks[j]));
            skip = rgb;
            snprintf(tn, sizeof tn, "r%d.rgb", r);
            reg(tn, rgb, {1, 3, r, r});

            x_cl = trunk_cl;  // stays channel-last, no transpose
        }


    }

    p.output("img", skip);  // [512,512,3] (truncated when TP_MDEC_STAGES<7)
    if (!B.err.empty()) {
        err = B.err;
        return false;
    }
    if (!p.commit(be, err)) return false;

    tap_list = std::move(taps);
    taps_in_graph = with_taps;
    built = true;
    return true;
}

bool motion_dec::set_feats(const menc_result &m, std::string &err) {
    if (!built) {
        err = "decoder not built";
        return false;
    }
    for (int j = 0; j < 7; j++) {
        char nm[16];
        snprintf(nm, sizeof nm, "feat%d", j);
        ggml_tensor *t = consts[nm];
        if (!t || (int64_t) m.feats[(size_t) j].size() != ggml_nelements(t)) {
            err = "feat size mismatch";
            return false;
        }
        if (t->type == GGML_TYPE_F16) {
            const std::vector<float> &src = m.feats[(size_t) j];
            std::vector<ggml_fp16_t> h(src.size());
            for (size_t i = 0; i < src.size(); i++) h[i] = ggml_fp32_to_fp16(src[i]);
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, m.feats[(size_t) j].data(), 0, ggml_nbytes(t));
        }
    }
    return true;
}

bool motion_dec::decode(backend_t &be, const float *style, float *img_out,
                        tap_sink &tap, std::string &err) {
    frame.set("style", style);
    if (!frame.compute(be, err)) return false;
    if (tap) {
        for (auto &e : tap_list)
            tap.pending.push_back({tap.prefix + e.name, e.t, e.shape});
        if (!tap.harvest()) {
            err = tap.err;
            return false;
        }
    }
    if (img_out) {
        ggml_tensor *img = frame["img"];
        ggml_backend_tensor_get(img, img_out, 0, ggml_nbytes(img));
    }
    return true;
}

void motion_dec::release() {
    frame.release();
    if (dbuf) ggml_backend_buffer_free(dbuf);
    if (dctx) ggml_free(dctx);
    if (cbuf) ggml_backend_buffer_free(cbuf);
    if (cctx) ggml_free(cctx);
    dbuf = nullptr;
    dctx = nullptr;
    cbuf = nullptr;
    cctx = nullptr;
    derived.clear();
    consts.clear();
    built = false;
}

}  // namespace tp
