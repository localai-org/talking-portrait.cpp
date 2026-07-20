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
        /*mem_size  =*/ ggml_tensor_overhead() * 96,
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
            // conv_transpose layout [KW, KH, OC, IC] for the spatial
            // direct-conv path (GGML_OP_CONV_TRANSPOSE_2D)
            ggml_tensor *tc = ggml_new_tensor_4d(dctx, wt, 3, 3, jb.oc, jb.ic);
            ggml_set_name(tc, ("tconv." + jb.name).c_str());
            derived["tconv." + jb.name] = tc;
            // sub-pixel composed kernel in spatial conv layout [3,3,IC,4OC]
            // (blur pre-composed; output needs the parity pixel-shuffle)
            ggml_tensor *tsp = ggml_new_tensor_4d(dctx, wt, 3, 3, jb.ic, 4 * jb.oc);
            ggml_set_name(tsp, ("tsubsp." + jb.name).c_str());
            derived["tsubsp." + jb.name] = tsp;
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
            // spatial layout of the same composed kernel: tsubsp[(q*3+kh)*3+kw
            // + ...] i.e. ggml [3,3,IC,4OC] index (kw,kh,i,q) = tb[q*R + i +
            // IC*(kw + 3*kh)]
            {
                std::vector<float> tsp((size_t) R * 4 * OC);
                for (int q = 0; q < 4 * OC; q++)
                    for (int i = 0; i < IC; i++)
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                                tsp[(((size_t) q * IC + i) * 3 + kh) * 3 + kw] =
                                    tb[(size_t) q * R + i + (size_t) IC * (kw + 3 * kh)];
                upload(derived["tsubsp." + jb.name], tsp);
            }
            // tconv[kw + 3kh + 9(o + OC i)] = W[o][i][kh][kw]
            std::vector<float> tc((size_t) IC * OC * 9);
            for (int o = 0; o < OC; o++)
                for (int i = 0; i < IC; i++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                            tc[(size_t) 9 * (o + (size_t) OC * i) + kh * 3 + kw] =
                                W[((size_t) o * IC + i) * 9 + kh * 3 + kw];
            upload(derived["tconv." + jb.name], tc);
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

// graph build ----------------------------------------------------------
//
// The fast (generation) graph is batched: BATCH frames per dispatch, all
// tensors carry the batch as their outermost dim (CL data [C, N, Bc]).
// Per-stage batch caps chunk the batch inside the graph to bound peak
// im2col memory at the big maps; chunk lists + views avoid any concat.

struct mdec_build {
    plan_t &p;
    ggml_context *ctx;
    weights_t *w;
    std::map<std::string, ggml_tensor *> &derived;
    ggml_tensor *latent = nullptr;  // [512, B]
    ggml_tensor *one = nullptr;     // [1, B] const 1.0
    bool fused = true;              // use the fused bias/lrelu/subpix ops
    std::string err;

    // Dataflow is CHANNEL-LAST ([C, N, Bc] GEMM outputs); spatial
    // [W,H,C,Bc] is materialized only where im2col / FIR / taps need it.

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
    // batch slice [*, Bc] of a full-batch [*, B] per-frame vector
    ggml_tensor *bstyle(ggml_tensor *t, int64_t c, int64_t Bc) {
        if (t->ne[1] == Bc && c == 0) return t;
        return ggml_view_2d(ctx, t, t->ne[0], Bc, t->nb[1], (size_t) c * Bc * t->nb[1]);
    }
    ggml_tensor *style_of(const std::string &base) {
        ggml_tensor *mw = need(base + ".modulation.weight");
        ggml_tensor *mb = need(base + ".modulation.bias");
        if (!mw || !mb) return nullptr;
        return linear(ctx, mw, mb, latent);  // [IC, B]
    }
    // demodulation denominator sqrt(S s^2 + eps) [OC, B]; S is pre-halved so
    // this also folds the FusedLeakyReLU sqrt(2)
    ggml_tensor *demod_den(const std::string &wname, ggml_tensor *style) {
        ggml_tensor *sa = der("saug." + wname);
        if (!sa) return nullptr;
        ggml_tensor *s2 = ggml_concat(ctx, ggml_sqr(ctx, style), one, 0);  // [IC+1,B]
        return ggml_sqrt(ctx, ggml_mul_mat(ctx, sa, s2));                  // [OC,B]
    }
    // GEMM A-matrix (rows kk-fastest) with optional per-in-channel style and
    // per-out-channel 1/den folded in; per-batch styles yield [KK*IC,OC,Bc]
    ggml_tensor *gemm_weight(ggml_tensor *W4d, ggml_tensor *style, ggml_tensor *den) {
        const int64_t KK = W4d->ne[0] * W4d->ne[1], IC = W4d->ne[2], OC = W4d->ne[3];
        ggml_tensor *A = ggml_reshape_2d(ctx, W4d, KK * IC, OC);
        if (!style && !den) return A;
        const int64_t Bc = style ? style->ne[1] : den->ne[1];
        const ggml_type t0 = A->type;
        ggml_tensor *Af = t0 == GGML_TYPE_F32 ? A : ggml_cast(ctx, A, GGML_TYPE_F32);
        if (Bc > 1) Af = ggml_repeat_4d(ctx, Af, KK * IC, OC, Bc, 1);
        if (style) {
            ggml_tensor *sty = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, style, 1, IC, Bc),
                                              KK, IC, Bc, 1);
            Af = ggml_mul(ctx, Af, ggml_reshape_3d(ctx, sty, KK * IC, 1, Bc));
        }
        if (den) Af = ggml_div(ctx, Af, ggml_reshape_3d(ctx, den, 1, OC, Bc));
        return t0 == GGML_TYPE_F32 ? Af : ggml_cast(ctx, Af, t0);
    }
    // channel-last [C, N, Bc] -> spatial [W, H, C, Bc]
    ggml_tensor *cl_to_sp(ggml_tensor *t_cl, int64_t Wd, int64_t Hd) {
        ggml_tensor *t = ggml_reshape_4d(ctx, t_cl, t_cl->ne[0], Wd, Hd,
                                         t_cl->ne[1] * t_cl->ne[2] / (Wd * Hd));
        return ggml_cont(ctx, ggml_permute(ctx, t, 2, 0, 1, 3));
    }
    // FusedLeakyReLU on channel-last (sqrt2 folded upstream); ab [1,1,C,1]
    ggml_tensor *bias_lrelu_cl(ggml_tensor *y_cl, ggml_tensor *ab) {
        if (fused) return ggml_bias_lrelu(ctx, y_cl, ab, 0, 0.2f);
        y_cl = ggml_add(ctx, y_cl, ggml_reshape_2d(ctx, ab, ab->ne[2], 1));
        return ggml_leaky_relu(ctx, y_cl, 0.2f, false);
    }
    // same on spatial [W,H,C,Bc] maps
    ggml_tensor *bias_lrelu_sp(ggml_tensor *y_sp, ggml_tensor *ab) {
        if (fused) return ggml_bias_lrelu(ctx, y_sp, ab, 2, 0.2f);
        return fused_lrelu(ctx, y_sp, ab);
    }
    // weight-side folding pays off only when the map is larger than the
    // weight matrix (big resolutions); small stages modulate the map
    static bool fold_into_weights(int64_t wsz, int64_t N, int64_t ic, int64_t oc) {
        return 4 * wsz < 2 * N * (ic + oc);
    }
    // fold style (per-in-channel, cf tiling) and 1/den (per-out-channel)
    // into a cf-ordered GEMM matrix A [R, OC'] -> [R, OC', Bc]
    ggml_tensor *fold_A(ggml_tensor *A, ggml_tensor *style, ggml_tensor *den) {
        if (!style && !den) return A;
        const int64_t R = A->ne[0], OCp = A->ne[1];
        const int64_t Bc = style ? style->ne[1] : den->ne[1];
        const ggml_type t0 = A->type;
        ggml_tensor *Af = t0 == GGML_TYPE_F32 ? A : ggml_cast(ctx, A, GGML_TYPE_F32);
        if (Bc > 1) Af = ggml_repeat_4d(ctx, Af, R, OCp, Bc, 1);
        if (style) {
            // cf rows are ic-fastest (r % IC = ic): plain tile along dim0
            ggml_tensor *sty = ggml_repeat_4d(ctx, style, R, Bc, 1, 1);
            Af = ggml_mul(ctx, Af, ggml_reshape_3d(ctx, sty, R, 1, Bc));
        }
        if (den) Af = ggml_div(ctx, Af, ggml_reshape_3d(ctx, den, 1, den->ne[0], Bc));
        return t0 == GGML_TYPE_F32 ? Af : ggml_cast(ctx, Af, t0);
    }
    // channels-first conv: x_cl [IC, N, Bc] over a WxH map. A2 [IC*9, OC']
    // carries the kernel dims; Af is A2 itself or a folded [IC*9, OC', Bc]
    ggml_tensor *conv_cf(ggml_tensor *A2, ggml_tensor *Af, ggml_tensor *x_cl,
                         int64_t W, int64_t H, int pad) {
        const int64_t IC = x_cl->ne[0], Bc = x_cl->ne[2];
        ggml_tensor *x4 = ggml_reshape_4d(ctx, x_cl, IC, W, H, Bc);
        ggml_tensor *shape4 = ggml_reshape_4d(ctx, A2, 3, 3, IC, A2->ne[1]);  // dims only
        ggml_tensor *im = ggml_im2col_cf(ctx, shape4, x4, 1, 1, pad, pad, 1, 1, A2->type);
        const int64_t N = im->ne[1] * im->ne[2];
        if (Af == A2 && Bc > 1) {
            // shared weight: flatten the batch into the GEMM N dim so the
            // A-matrix is streamed once, not once per batch element
            ggml_tensor *y = ggml_mul_mat(ctx, Af, ggml_reshape_2d(ctx, im, im->ne[0], N * Bc));
            return ggml_reshape_3d(ctx, y, y->ne[0], N, Bc);
        }
        return ggml_mul_mat(ctx, Af, ggml_reshape_3d(ctx, im, im->ne[0], N, Bc));
    }
    // normal modulated conv 3x3 (demodulate) + FLReLU: CL in, CL out.
    // style/den are full-batch [., B]; c/Bc select the chunk.
    ggml_tensor *styled_conv(ggml_tensor *x_cl, int64_t W, int64_t H, const std::string &base,
                             ggml_tensor *style, ggml_tensor *den, int64_t c, int64_t Bc) {
        ggml_tensor *A = der("acf." + base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !den || !A || !ab) return nullptr;
        ggml_tensor *sty = bstyle(style, c, Bc), *dn = bstyle(den, c, Bc);
        const int64_t N = W * H, IC = x_cl->ne[0];
        ggml_tensor *y;
        if (fold_into_weights(A->ne[0] * A->ne[1], N, IC, A->ne[1])) {
            y = conv_cf(A, fold_A(A, sty, dn), x_cl, W, H, 1);
        } else {
            y = conv_cf(A, A,
                        ggml_mul(ctx, x_cl, ggml_reshape_3d(ctx, sty, IC, 1, Bc)), W, H, 1);
            y = ggml_div(ctx, y, ggml_reshape_3d(ctx, dn, dn->ne[0], 1, Bc));
        }
        return bias_lrelu_cl(y, ab);
    }
    // first 4x4 styled conv (spatial const input, tiny; full batch)
    ggml_tensor *styled_conv_input(ggml_tensor *x_sp, const std::string &base,
                                   ggml_tensor *style, ggml_tensor *den) {
        ggml_tensor *W = need(base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!W || !ab || !style || !den) return nullptr;
        const int64_t C = style->ne[0], Bb = style->ne[1];
        ggml_tensor *x = x_sp;
        if (Bb > 1)
            x = ggml_repeat_4d(ctx, x_sp, x_sp->ne[0], x_sp->ne[1], x_sp->ne[2], Bb);
        ggml_tensor *xs = ggml_mul(ctx, x, ggml_reshape_4d(ctx, style, 1, 1, C, Bb));
        ggml_tensor *im = ggml_im2col(ctx, W, xs, 1, 1, 1, 1, 1, 1, true, W->type);
        ggml_tensor *Am = ggml_reshape_2d(ctx, W, W->ne[0] * W->ne[1] * W->ne[2], W->ne[3]);
        const int64_t NN = im->ne[1] * im->ne[2];
        // shared weight: flatten batch into the GEMM N dim (stream A once)
        ggml_tensor *y = ggml_mul_mat(ctx, Am, ggml_reshape_2d(ctx, im, im->ne[0], NN * Bb));
        y = ggml_reshape_3d(ctx, y, y->ne[0], NN, Bb);
        y = ggml_div(ctx, y, ggml_reshape_3d(ctx, den, den->ne[0], 1, Bb));
        return bias_lrelu_cl(y, ab);  // [OC, 16, B]
    }
    // upsampling modulated conv: sub-pixel form of conv_transpose(s2)+Blur —
    // one 3x3 pad-1 conv, 4 parity classes stacked on OC, interleaved in
    // channel-last; batch rides on ne3 through the interleave
    ggml_tensor *styled_conv_up(ggml_tensor *x_cl, int64_t W, int64_t H, const std::string &base,
                                ggml_tensor *style, ggml_tensor *den, int64_t c, int64_t Bc) {
        ggml_tensor *Ts = der("tsubcf." + base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !den || !Ts || !ab) return nullptr;
        const int64_t OC = Ts->ne[1] / 4, IC = x_cl->ne[0];
        const int64_t N = W * H;
        ggml_tensor *sty = bstyle(style, c, Bc), *dn = bstyle(den, c, Bc);
        // per-parity denominator: column q = k*OC + o, q % OC = o -> tile
        ggml_tensor *den4 = ggml_repeat_4d(ctx, dn, 4 * OC, Bc, 1, 1);
        ggml_tensor *y4;
        if (fold_into_weights(Ts->ne[0] * Ts->ne[1], N, IC, Ts->ne[1])) {
            y4 = conv_cf(Ts, fold_A(Ts, sty, den4), x_cl, W, H, 1);
        } else {
            y4 = conv_cf(Ts, Ts,
                         ggml_mul(ctx, x_cl, ggml_reshape_3d(ctx, sty, IC, 1, Bc)), W, H, 1);
            y4 = ggml_div(ctx, y4, ggml_reshape_3d(ctx, den4, 4 * OC, 1, Bc));
        }
        ggml_tensor *cls[4];
        for (int k = 0; k < 4; k++)
            cls[k] = ggml_cont(ctx, ggml_view_3d(ctx, y4, OC, N, Bc, y4->nb[1], y4->nb[2],
                                                 (size_t) k * OC * y4->nb[0]));
        // u-interleave: (o, bx, m) -> flat o + OC*bx + 2OC*m
        ggml_tensor *E[2];
        for (int by = 0; by < 2; by++) {
            ggml_tensor *a = ggml_reshape_4d(ctx, cls[by * 2 + 0], OC, 1, N, Bc);
            ggml_tensor *b = ggml_reshape_4d(ctx, cls[by * 2 + 1], OC, 1, N, Bc);
            // [OC,2,N,Bc] == flat o + OC*bx + 2OC*(x + W*y) -> [(o,u), y, b]
            E[by] = ggml_reshape_3d(ctx, ggml_concat(ctx, a, b, 1), OC * 2 * W, H, Bc);
        }
        // v-interleave: (q, by, y) -> flat q + 2OCW*by + 4OCW*y
        ggml_tensor *F = ggml_concat(ctx, ggml_reshape_4d(ctx, E[0], OC * 2 * W, 1, H, Bc),
                                     ggml_reshape_4d(ctx, E[1], OC * 2 * W, 1, H, Bc), 1);
        F = ggml_reshape_3d(ctx, F, OC, 4 * N, Bc);  // CL over the 2W x 2H map
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
    // ---- spatial direct-conv fast-stage variants (big maps): implicit-GEMM
    // CONV_2D / CONV_TRANSPOSE_2D, style+den folded into the (tiny) conv
    // weights per frame — no im2col materialization, no full-map style/den
    // passes. Chunk size is 1 (weights are per frame). ----
    // W4 [KW,KH,IC,OC] (conv) or [KW,KH,OC,IC] (conv_transpose, tl=true).
    // The folded weight is cast back to the source dtype: the f16 weight
    // path keeps CONV_2D on the (coopmat2-staged) f16 pipelines — the f32
    // conv2d pipeline corrupts some shapes here (see docs/vulkan-perf.md).
    ggml_tensor *fold_w4(ggml_tensor *W4, ggml_tensor *style, ggml_tensor *den, bool tl) {
        const ggml_type t0 = W4->type;
        ggml_tensor *Wf = t0 == GGML_TYPE_F32 ? W4 : ggml_cast(ctx, W4, GGML_TYPE_F32);
        if (style) {
            ggml_tensor *s4 = tl ? ggml_reshape_4d(ctx, style, 1, 1, 1, style->ne[0])
                                 : ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1);
            Wf = ggml_mul(ctx, Wf, s4);
        }
        if (den) {
            ggml_tensor *d4 = tl ? ggml_reshape_4d(ctx, den, 1, 1, den->ne[0], 1)
                                 : ggml_reshape_4d(ctx, den, 1, 1, 1, den->ne[0]);
            Wf = ggml_div(ctx, Wf, d4);
        }
        return t0 == GGML_TYPE_F32 ? Wf : ggml_cast(ctx, Wf, t0);
    }
    // modulated 3x3 conv (demodulate) + FLReLU, spatial in/out, frame c
    ggml_tensor *styled_conv_sp(ggml_tensor *x_sp, const std::string &base,
                                ggml_tensor *style, ggml_tensor *den, int64_t c) {
        ggml_tensor *W = need(base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!W || !ab || !style || !den) return nullptr;
        ggml_tensor *Wf = fold_w4(W, bstyle(style, c, 1), bstyle(den, c, 1), false);
        ggml_tensor *y = ggml_conv_2d_direct(ctx, Wf, x_sp, 1, 1, 1, 1, 1, 1);
        return bias_lrelu_sp(y, ab);
    }
    // modulated up conv, spatial: composed sub-pixel kernel (blur folded in)
    // as one 3x3 pad-1 conv producing 4 parity classes on the channel dim,
    // then a spatial pixel-shuffle. Frame c (chunk size 1).
    ggml_tensor *styled_conv_up_sp(ggml_tensor *x_sp, const std::string &base,
                                   ggml_tensor *style, ggml_tensor *den, int64_t c) {
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!ab || !style || !den) return nullptr;
        ggml_tensor *sty = bstyle(style, c, 1), *dn = bstyle(den, c, 1);
        if (getenv("TP_MDEC_UPSP_TCONV")) {
            // A/B variant: conv_transpose + separate blur pass
            ggml_tensor *Tc = der("tconv." + base + ".conv.weight");
            ggml_tensor *kb = need(base + ".conv.blur.kernel");
            if (!Tc || !kb) return nullptr;
            const int OC = (int) Tc->ne[2];
            ggml_tensor *Wf = fold_w4(Tc, sty, dn, true);
            ggml_tensor *y = ggml_conv_transpose_2d_p0(ctx, Wf, x_sp, 2);
            ggml_tensor *k4 =
                ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, kb, 4, 4, 1, 1), 4, 4, 1, OC);
            y = blur(ctx, y, k4, 1, 1);  // [2W, 2H, OC]
            return fused_lrelu(ctx, y, ab);
        }
        ggml_tensor *Ts = der("tsubsp." + base + ".conv.weight");
        if (!Ts) return nullptr;
        const int64_t OC = Ts->ne[3] / 4;
        const int64_t W = x_sp->ne[0], H = x_sp->ne[1];
        // per-parity denominator: channel q = k*OC + o, q % OC = o -> tile
        ggml_tensor *den4 = ggml_repeat_4d(ctx, dn, 4 * OC, 1, 1, 1);
        if (fused && !getenv("TP_MDEC_UPSP_MAPSIDE") && !getenv("TP_MDEC_UPSP_CLCONV")) {
            // two half convs (one per row parity; wide-Cout CONV_2D corrupts
            // on this driver) + fused interleave/bias/lrelu
            ggml_tensor *Wf = fold_w4(Ts, sty, den4, false);  // [3,3,IC,4OC]
            ggml_tensor *grp[2];
            for (int g = 0; g < 2; g++) {
                ggml_tensor *Wk = ggml_view_4d(ctx, Wf, Wf->ne[0], Wf->ne[1], Wf->ne[2],
                                               2 * OC, Wf->nb[1], Wf->nb[2], Wf->nb[3],
                                               (size_t) g * 2 * OC * Wf->nb[3]);
                grp[g] = ggml_conv_2d_direct(ctx, Wk, x_sp, 1, 1, 1, 1, 1, 1);
            }
            return ggml_subpix_lrelu(ctx, grp[0], grp[1], ab, 0.2f);
        }
        ggml_tensor *y4;
        if (getenv("TP_MDEC_UPSP_MAPSIDE")) {
            // debug: modulate the map, conv with the leaf weight tensor
            ggml_tensor *xs = ggml_mul(
                ctx, x_sp, ggml_reshape_4d(ctx, sty, 1, 1, sty->ne[0], 1));
            y4 = ggml_conv_2d_direct(ctx, Ts, xs, 1, 1, 1, 1, 1, 1);
            y4 = ggml_div(ctx, y4, ggml_reshape_4d(ctx, den4, 1, 1, 4 * OC, 1));
        } else if (getenv("TP_MDEC_UPSP_CLCONV")) {
            // debug: conv via the CL im2col+GEMM path, then to spatial
            ggml_tensor *Tcf = der("tsubcf." + base + ".conv.weight");
            if (!Tcf) return nullptr;
            ggml_tensor *x_cl3 = ggml_reshape_3d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, x_sp, 1, 2, 0, 3)),
                x_sp->ne[2], W * H, 1);
            ggml_tensor *y_cl = conv_cf(Tcf, fold_A(Tcf, sty, den4), x_cl3, W, H, 1);
            y4 = cl_to_sp(y_cl, W, H);  // [W, H, 4OC, 1]
        } else {
            y4 = nullptr;  // per-class convs below (see cls)
        }
        // pixel shuffle: out[2n+bx, 2m+by, o] = class k = 2by+bx at [n, m, o]
        ggml_tensor *cls[4];
        if (y4) {
            for (int k = 0; k < 4; k++)
                cls[k] = ggml_cont(ctx, ggml_view_3d(ctx, y4, W, H, OC, y4->nb[1],
                                                     y4->nb[2], (size_t) k * OC * y4->nb[2]));
        } else {
            // split the 4-class conv: wide-Cout CONV_2D shapes corrupt on
            // this driver (cf. docs/vulkan-perf.md). TP_MDEC_UPSPLIT sets
            // convs per up (1|2|4), csv per spatial stage from stage 5.
            int nsplit = 4;
            if (const char *e = getenv("TP_MDEC_UPSPLIT")) {
                // pick the entry matching this stage's OC (64->j6:2nd, ...)
                int v1 = 4, v2 = 4;
                sscanf(e, "%d,%d", &v1, &v2);
                nsplit = (x_sp->ne[2] == 64) ? v2 : v1;  // IC 64 = stage 6
                if (nsplit != 1 && nsplit != 2) nsplit = 4;
            }
            ggml_tensor *Wf = fold_w4(Ts, sty, den4, false);  // [3,3,IC,4OC]
            const int64_t OCg = 4 * OC / nsplit;
            ggml_tensor *grp[4];
            for (int g = 0; g < nsplit; g++) {
                ggml_tensor *Wk = ggml_view_4d(ctx, Wf, Wf->ne[0], Wf->ne[1], Wf->ne[2],
                                               OCg, Wf->nb[1], Wf->nb[2], Wf->nb[3],
                                               (size_t) g * OCg * Wf->nb[3]);
                grp[g] = ggml_conv_2d_direct(ctx, Wk, x_sp, 1, 1, 1, 1, 1, 1);
            }
            const int64_t per = 4 / nsplit;  // classes per group
            for (int k = 0; k < 4; k++) {
                ggml_tensor *gt = grp[k / per];
                if (per == 1) {
                    cls[k] = gt;
                } else {
                    cls[k] = ggml_cont(
                        ctx, ggml_view_3d(ctx, gt, W, H, OC, gt->nb[1], gt->nb[2],
                                          (size_t) (k % per) * OC * gt->nb[2]));
                }
            }
        }
        ggml_tensor *E[2];
        for (int by = 0; by < 2; by++) {
            ggml_tensor *a = ggml_reshape_4d(ctx, cls[by * 2 + 0], 1, W, H, OC);
            ggml_tensor *b = ggml_reshape_4d(ctx, cls[by * 2 + 1], 1, W, H, OC);
            E[by] = ggml_reshape_3d(ctx, ggml_concat(ctx, a, b, 0), 2 * W, H, OC);
        }
        ggml_tensor *F = ggml_concat(ctx, ggml_reshape_4d(ctx, E[0], 2 * W, 1, H, OC),
                                     ggml_reshape_4d(ctx, E[1], 2 * W, 1, H, OC), 1);
        F = ggml_reshape_4d(ctx, F, 2 * W, 2 * H, OC, 1);
        return fused_lrelu(ctx, F, ab);
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

    // bilinear zeros-padding grid_sample of feat_t [C, r*r] at grid points
    // gx/gy [r, r, Bc] (feat shared across the batch): returns CL [C, r*r, Bc]
    ggml_tensor *grid_sample_cl(ggml_tensor *feat_t, ggml_tensor *gx, ggml_tensor *gy, int r) {
        const int64_t Bc = gx->ne[2];
        const int64_t NN = (int64_t) r * r * Bc;
        ggml_tensor *grid = ggml_concat(ctx, ggml_reshape_2d(ctx, gx, 1, NN),
                                        ggml_reshape_2d(ctx, gy, 1, NN), 0);  // [2, N*Bc]
        ggml_tensor *s = ggml_bilinear_gather(ctx, feat_t, grid, r, r);
        return ggml_reshape_3d(ctx, s, s->ne[0], (int64_t) r * r, Bc);
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
        mk("ones", 1, MAX_BATCH);
        cbuf = ggml_backend_alloc_ctx_tensors(cctx, be.be);
        if (!cbuf) {
            err = "failed to allocate mdec const buffer";
            return false;
        }
        // fill masks / identity grids / ones
        std::vector<float> buf;
        {
            const std::vector<float> ones((size_t) MAX_BATCH, 1.0f);
            ggml_backend_tensor_set(consts["ones"], ones.data(), 0, MAX_BATCH * 4);
        }
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
    onec = consts["ones"];

    // TP_MDEC_EXACT=1 forces the conservative graph for generation too:
    // measurement knob for backends where the fast graph's custom ops fall
    // back to the CPU (e.g. Vulkan). Taps stay tied to with_taps.
    const bool exact_graph = with_taps || getenv("TP_MDEC_EXACT") != nullptr;

    // fast-graph batch: frames per graph dispatch (ship default 8)
    int bsz = 1;
    if (!exact_graph) {
        bsz = 8;
        if (const char *e = getenv("TP_MDEC_BATCH")) bsz = atoi(e);
        if (bsz < 1) bsz = 1;
        if (bsz > MAX_BATCH) bsz = MAX_BATCH;
        // chunk caps require power-of-two divisibility
        while (bsz & (bsz - 1)) bsz--;
    }
    batch = bsz;

    plan_t &p = frame;
    p.begin(4096);
    ggml_context *ctx = p.ctx;
    mdec_build B{p, ctx, w, derived};
    B.fused = !exact_graph && getenv("TP_MDEC_NOFUSED") == nullptr;

    B.latent = p.input("style", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, bsz));
    B.one = ggml_view_2d(ctx, onec, 1, bsz, onec->nb[1], 0);

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
        p.output("img", skip);  // [512,512,3] (truncated when TP_MDEC_STAGES<7)
        img_chunks = 0;
    } else {
        // ---- batched fast graph: bsz frames per dispatch. Per-stage batch
        // caps chunk the batch inside the graph so peak im2col memory at the
        // big maps stays at the per-frame level; chunk lists sliced by views
        // keep the whole thing concat-free. ----
        struct bchunks {
            std::vector<ggml_tensor *> t;  // equal batch-size chunks
            int64_t bc = 0;                // frames per chunk
        };
        // slice Bc frames starting at c*Bc out of a CL [C, N, bc] chunk list
        auto cl_slice = [&](bchunks &l, int64_t c, int64_t Bc) -> ggml_tensor * {
            ggml_tensor *src = l.t[(size_t) (c * Bc / l.bc)];
            if (Bc == l.bc) return src;
            const int64_t off = (c * Bc) % l.bc;
            return ggml_view_3d(ctx, src, src->ne[0], src->ne[1], Bc, src->nb[1],
                                src->nb[2], (size_t) off * src->nb[2]);
        };
        // same for spatial [W, H, C, bc] chunk lists
        auto sp_slice = [&](bchunks &l, int64_t c, int64_t Bc) -> ggml_tensor * {
            ggml_tensor *src = l.t[(size_t) (c * Bc / l.bc)];
            if (Bc == l.bc) return src;
            const int64_t off = (c * Bc) % l.bc;
            return ggml_view_4d(ctx, src, src->ne[0], src->ne[1], src->ne[2], Bc,
                                src->nb[1], src->nb[2], src->nb[3],
                                (size_t) off * src->nb[3]);
        };
        // channel plane [r, r, Bc] out of a flowmap [r, r, 3, Bc]
        auto flch = [&](ggml_tensor *fl, int chn) {
            return ggml_cont(ctx, ggml_view_3d(ctx, fl, fl->ne[0], fl->ne[1], fl->ne[3],
                                               fl->nb[1], fl->nb[3],
                                               (size_t) chn * fl->nb[2]));
        };

        // stages >= spatial_from use the spatial direct-conv path (chunk 1);
        // env TP_MDEC_SPATIAL overrides (7 = none)
        int spatial_from = 4;
        if (const char *e = getenv("TP_MDEC_SPATIAL")) {
            spatial_from = atoi(e);
            if (spatial_from < 0 || spatial_from > 7) spatial_from = 7;
        }
        // per-stage batch caps (peak-VRAM guard); env TP_MDEC_BMAX=csv, 0=full
        int64_t cap[7] = {bsz, bsz, bsz, std::min<int64_t>(bsz, 4), bsz,
                          std::min<int64_t>(bsz, 2), std::min<int64_t>(bsz, 1)};
        if (const char *e = getenv("TP_MDEC_BMAX")) {
            char tmp[128];
            snprintf(tmp, sizeof tmp, "%s", e);
            char *save = nullptr;
            int idx = 0;
            for (char *tok = strtok_r(tmp, ",", &save); tok && idx < 7;
                 tok = strtok_r(nullptr, ",", &save), idx++) {
                const int v = atoi(tok);
                cap[idx] = v < 1 ? bsz : v;
            }
        }
        for (int j = 0; j < 7; j++) {
            if (j >= spatial_from) cap[j] = 1;  // per-frame folded weights
            if (cap[j] > bsz) cap[j] = bsz;
            while (cap[j] > 1 && bsz % cap[j]) cap[j]--;   // must divide bsz
            if (j > 0 && cap[j] > cap[j - 1]) cap[j] = cap[j - 1];  // monotone
        }

        // constant input + conv1 + (no to_rgb1 in forward)
        ggml_tensor *cinp = B.need("dec.input.input");  // [4,4,512,1]
        ggml_tensor *s0 = B.style_of("dec.conv1.conv");
        ggml_tensor *d0 = s0 ? B.demod_den("dec.conv1.conv.weight", s0) : nullptr;
        ggml_tensor *x0 =
            (cinp && s0 && d0) ? B.styled_conv_input(cinp, "dec.conv1", s0, d0) : nullptr;
        if (!x0) {
            err = B.err.empty() ? "conv1 failed" : B.err;
            return false;
        }
        bchunks x{{x0}, bsz};    // trunk, CL [C, N, bc] (or spatial)
        bool x_is_sp = false;    // trunk layout: CL vs spatial [W,H,C,bc]
        bchunks sk{{}, bsz};     // rgb accumulator, spatial [r, r, 3, bc]
        bchunks skf{{}, bsz};    // flow accumulator, spatial [r, r, 3, bc]

        for (int j = 0; j < max_stage; j++) {
            const int r = 8 << j;
            const int64_t N = (int64_t) r * r;
            const int64_t Bc = cap[j], nch = bsz / Bc;
            char b1[32], b2[32], bf[32], br[32];
            snprintf(b1, sizeof b1, "dec.convs.%d", 2 * j);
            snprintf(b2, sizeof b2, "dec.convs.%d", 2 * j + 1);
            snprintf(bf, sizeof bf, "dec.to_flows.%d", j);
            snprintf(br, sizeof br, "dec.to_rgbs.%d", j);
            const std::string fbase = bf, rbase = br;

            // stage styles: one batched computation, sliced per chunk
            ggml_tensor *s1 = B.style_of(std::string(b1) + ".conv");
            ggml_tensor *d1 = s1 ? B.demod_den(std::string(b1) + ".conv.weight", s1) : nullptr;
            ggml_tensor *s2 = B.style_of(std::string(b2) + ".conv");
            ggml_tensor *d2 = s2 ? B.demod_den(std::string(b2) + ".conv.weight", s2) : nullptr;
            ggml_tensor *fstyle = B.style_of(fbase + ".conv");
            ggml_tensor *fW = B.need(fbase + ".conv.weight");
            ggml_tensor *fbias = B.need(fbase + ".bias");
            ggml_tensor *fup = B.need(fbase + ".upsample.kernel");
            ggml_tensor *rW = B.need(rbase + ".conv.0.weight");
            ggml_tensor *rlb = B.need(rbase + ".conv.1.bias");
            ggml_tensor *rbias = B.need(rbase + ".bias");
            ggml_tensor *rup = B.need(rbase + ".upsample.kernel");
            if (!s1 || !d1 || !s2 || !d2 || !fstyle || !fW || !fbias || !fup || !rW ||
                !rlb || !rbias || !rup) {
                err = B.err;
                return false;
            }

            const bool spatial = j >= spatial_from;
            bchunks xn{{}, Bc}, skn{{}, Bc}, sfn{{}, Bc};
            for (int64_t c = 0; c < nch; c++) {
                ggml_tensor *fl, *m;
                if (spatial) {
                    // ---- spatial direct-conv stage (Bc == 1) ----
                    ggml_tensor *x_c = x_is_sp
                        ? sp_slice(x, c, 1)
                        : B.cl_to_sp(cl_slice(x, c, 1), r / 2, r / 2);
                    ggml_tensor *c1 = B.styled_conv_up_sp(x_c, b1, s1, d1, c);
                    ggml_tensor *out =
                        c1 ? B.styled_conv_sp(c1, b2, s2, d2, c) : nullptr;
                    if (!out) {
                        err = B.err;
                        return false;
                    }
                    // ToFlow: 1x1 modulated conv, weight-folded
                    ggml_tensor *fWf =
                        B.fold_w4(fW, B.bstyle(fstyle, c, 1), nullptr, false);
                    fl = ggml_conv_2d_direct(ctx, fWf, out, 1, 1, 0, 0, 1, 1);
                    fl = ggml_add(ctx, fl, fbias);
                    if (!skf.t.empty())
                        fl = ggml_add(ctx, fl,
                                      B.upsample2x(sp_slice(skf, c, 1), fup, masks[j]));
                    sfn.t.push_back(fl);

                    ggml_tensor *gx = ggml_add(ctx, ggml_tanh(ctx, flch(fl, 0)), xs_x[j]);
                    ggml_tensor *gy = ggml_add(ctx, ggml_tanh(ctx, flch(fl, 1)), xs_y[j]);
                    m = ggml_sigmoid(ctx, flch(fl, 2));  // [r,r,1]
                    ggml_tensor *m4 = ggml_reshape_4d(ctx, m, r, r, 1, 1);
                    ggml_tensor *sampled_sp = B.cl_to_sp(
                        B.grid_sample_cl(feats[j], gx, gy, r), r, r);  // [r,r,C,1]
                    ggml_tensor *warp = ggml_mul(ctx, sampled_sp, m4);
                    ggml_tensor *one_m = ggml_reshape_4d(
                        ctx, ggml_scale_bias(ctx, m, -1.0f, 1.0f), r, r, 1, 1);
                    ggml_tensor *trunk =
                        ggml_add(ctx, warp, ggml_mul(ctx, out, one_m));
                    xn.t.push_back(trunk);

                    // ToRGB: 1x1 EqualConv2d direct
                    ggml_tensor *rgb = ggml_conv_2d_direct(ctx, rW, warp, 1, 1, 0, 0, 1, 1);
                    rgb = B.bias_lrelu_sp(rgb, rlb);
                    rgb = ggml_add(ctx, rgb, rbias);
                    if (!sk.t.empty())
                        rgb = ggml_add(ctx, rgb,
                                       B.upsample2x(sp_slice(sk, c, 1), rup, masks[j]));
                    skn.t.push_back(rgb);
                    continue;
                }
                ggml_tensor *x_c = cl_slice(x, c, Bc);
                ggml_tensor *c1_cl =
                    B.styled_conv_up(x_c, r / 2, r / 2, b1, s1, d1, c, Bc);  // [OC,N,Bc]
                ggml_tensor *out_cl =
                    c1_cl ? B.styled_conv(c1_cl, r, r, b2, s2, d2, c, Bc) : nullptr;
                if (!out_cl) {
                    err = B.err;
                    return false;
                }

                // ToFlow: 1x1 modulated conv == batched GEMM on the CL tensor
                ggml_tensor *fl_cl = ggml_mul_mat(
                    ctx, B.gemm_weight(fW, B.bstyle(fstyle, c, Bc), nullptr), out_cl);
                fl_cl = ggml_add(ctx, fl_cl, ggml_reshape_2d(ctx, fbias, 3, 1));
                fl = B.cl_to_sp(fl_cl, r, r);  // [r,r,3,Bc]
                if (!skf.t.empty())
                    fl = ggml_add(ctx, fl,
                                  B.upsample2x(sp_slice(skf, c, Bc), fup, masks[j]));
                sfn.t.push_back(fl);

                ggml_tensor *gx = ggml_add(ctx, ggml_tanh(ctx, flch(fl, 0)), xs_x[j]);
                ggml_tensor *gy = ggml_add(ctx, ggml_tanh(ctx, flch(fl, 1)), xs_y[j]);
                m = ggml_sigmoid(ctx, flch(fl, 2));  // [r,r,Bc]
                ggml_tensor *m_row = ggml_reshape_3d(ctx, m, 1, N, Bc);
                ggml_tensor *sampled = B.grid_sample_cl(feats[j], gx, gy, r);  // [C,N,Bc]
                ggml_tensor *warp_cl = ggml_mul(ctx, sampled, m_row);
                ggml_tensor *one_m_row = ggml_reshape_3d(
                    ctx, ggml_scale_bias(ctx, m, -1.0f, 1.0f), 1, N, Bc);
                ggml_tensor *trunk_cl =
                    ggml_add(ctx, warp_cl, ggml_mul(ctx, out_cl, one_m_row));
                xn.t.push_back(trunk_cl);

                // ToRGB(out_warp, skip): 1x1 EqualConv2d == GEMM on warp_cl
                ggml_tensor *rgb_cl =
                    ggml_mul_mat(ctx, B.gemm_weight(rW, nullptr, nullptr), warp_cl);
                rgb_cl = B.bias_lrelu_cl(rgb_cl, rlb);
                rgb_cl = ggml_add(ctx, rgb_cl, ggml_reshape_2d(ctx, rbias, 3, 1));
                ggml_tensor *rgb = B.cl_to_sp(rgb_cl, r, r);
                if (!sk.t.empty())
                    rgb = ggml_add(ctx, rgb,
                                   B.upsample2x(sp_slice(sk, c, Bc), rup, masks[j]));
                skn.t.push_back(rgb);
            }
            x = std::move(xn);
            sk = std::move(skn);
            skf = std::move(sfn);
            if (spatial) x_is_sp = true;
        }

        // one output per final chunk: frames [i*bc, (i+1)*bc)
        for (size_t i = 0; i < sk.t.size(); i++) {
            char nm[16];
            snprintf(nm, sizeof nm, "img%zu", i);
            p.output(nm, sk.t[i]);
        }
        img_chunks = (int) sk.t.size();
    }

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
    if (img_chunks > 0) {
        // fast batched plan: replicate the style across the batch, keep frame 0
        std::vector<float> styles((size_t) batch * 512);
        for (int b = 0; b < batch; b++) memcpy(styles.data() + (size_t) b * 512, style, 512 * 4);
        frame.set("style", styles.data());
        if (!frame.compute(be, err)) return false;
        if (img_out) {
            ggml_tensor *img = frame["img0"];
            const size_t fsz = ggml_nbytes(img) / (size_t) img->ne[3];
            ggml_backend_tensor_get(img, img_out, 0, fsz);
        }
        return true;
    }
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

bool motion_dec::decode_batch(backend_t &be, const float *styles, float *imgs_out,
                              std::string &err) {
    if (img_chunks == 0) {
        // exact plan is batch-1: decode frame by frame
        tap_sink no_tap;
        for (int b = 0; b < batch; b++) {
            const size_t fsz = (size_t) 3 * 512 * 512;
            if (!decode(be, styles + (size_t) b * 512,
                        imgs_out ? imgs_out + (size_t) b * fsz : nullptr, no_tap, err))
                return false;
        }
        return true;
    }
    frame.set("style", styles);
    if (!frame.compute(be, err)) return false;
    if (imgs_out) {
        size_t off = 0;
        for (int i = 0; i < img_chunks; i++) {
            char nm[16];
            snprintf(nm, sizeof nm, "img%d", i);
            ggml_tensor *img = frame[nm];
            ggml_backend_tensor_get(img, imgs_out + off, 0, ggml_nbytes(img));
            off += (size_t) ggml_nelements(img);
        }
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
