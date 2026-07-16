#include "motion_dec.h"

#include "ops_common.h"

#include <cmath>
#include <cstdio>

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
        if (jb.flip) {
            ggml_tensor *wf = ggml_new_tensor_4d(dctx, GGML_TYPE_F32, 3, 3, jb.ic, jb.oc);
            ggml_set_name(wf, ("wflip." + jb.name).c_str());
            derived["wflip." + jb.name] = wf;
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
        buf.assign((size_t) (IC + 1) * OC, 0.0f);
        for (int o = 0; o < OC; o++) {
            for (int i = 0; i < IC; i++) {
                double s = 0;
                const float *pw = W.data() + ((size_t) o * IC + i) * 9;
                for (int k = 0; k < 9; k++) s += (double) pw[k] * pw[k];
                buf[(size_t) o * (IC + 1) + i] = (float) s;
            }
            buf[(size_t) o * (IC + 1) + IC] = 1e-8f;
        }
        ggml_tensor *sa = derived["saug." + jb.name];
        ggml_backend_tensor_set(sa, buf.data(), 0, ggml_nbytes(sa));
        if (jb.flip) {
            buf.assign(W.size(), 0.0f);
            for (int o = 0; o < OC; o++)
                for (int i = 0; i < IC; i++)
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            buf[(((size_t) o * IC + i) * 3 + a) * 3 + b] =
                                W[(((size_t) o * IC + i) * 3 + (2 - a)) * 3 + (2 - b)];
            ggml_tensor *wf = derived["wflip." + jb.name];
            ggml_backend_tensor_set(wf, buf.data(), 0, ggml_nbytes(wf));
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
    ggml_tensor *demod_div(ggml_tensor *conv_out, const std::string &wname, ggml_tensor *style) {
        ggml_tensor *sa = der("saug." + wname);
        if (!sa) return conv_out;
        ggml_tensor *s2 = ggml_concat(ctx, ggml_sqr(ctx, style), one, 0);  // [IC+1,1]
        ggml_tensor *den = ggml_sqrt(ctx, ggml_mul_mat(ctx, sa, s2));      // [OC,1]
        den = ggml_reshape_4d(ctx, den, 1, 1, den->ne[0], 1);
        return ggml_div(ctx, conv_out, den);
    }
    // normal modulated conv 3x3 (demodulate) + FusedLeakyReLU
    ggml_tensor *styled_conv(ggml_tensor *x, const std::string &base) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *W = need(base + ".conv.weight");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !W || !ab) return nullptr;
        ggml_tensor *xs = ggml_mul(ctx, x, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        ggml_tensor *y = conv2d(ctx, W, xs, 1, 1);
        y = demod_div(y, base + ".conv.weight", style);
        return fused_lrelu(ctx, y, ab);
    }
    // upsampling modulated conv (conv_transpose stride 2 + blur) + FLReLU
    ggml_tensor *styled_conv_up(ggml_tensor *x, const std::string &base, ggml_tensor *mask2x) {
        ggml_tensor *style = style_of(base + ".conv");
        ggml_tensor *Wf = der("wflip." + base + ".conv.weight");
        ggml_tensor *kb = need(base + ".conv.blur.kernel");
        ggml_tensor *ab = need(base + ".activate.bias");
        if (!style || !Wf || !kb || !ab) return nullptr;
        const int OC = (int) Wf->ne[3];
        ggml_tensor *xs = ggml_mul(ctx, x, ggml_reshape_4d(ctx, style, 1, 1, style->ne[0], 1));
        ggml_tensor *zs = zero_stuff2(ctx, xs, mask2x);            // [2W,2W,IC]
        zs = ggml_pad_ext(ctx, zs, 2, 1, 2, 1, 0, 0, 0, 0);
        ggml_tensor *y = conv2d(ctx, Wf, zs, 1, 0);                // [2W+1,2W+1,OC]
        y = demod_div(y, base + ".conv.weight", style);
        ggml_tensor *k4 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, kb, 4, 4, 1, 1), 4, 4, 1, OC);
        y = blur(ctx, y, k4, 1, 1);                                // [2W,2W,OC]
        return fused_lrelu(ctx, y, ab);
    }
    // upfirdn2d up=2 pad(2,1) with a x4 kernel (Upsample module)
    ggml_tensor *upsample2x(ggml_tensor *x, ggml_tensor *kern, ggml_tensor *mask2x) {
        const int C = (int) x->ne[2];
        ggml_tensor *zs = zero_stuff2(ctx, x, mask2x);
        zs = ggml_pad_ext(ctx, zs, 2, 1, 2, 1, 0, 0, 0, 0);
        ggml_tensor *k4 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, kern, 4, 4, 1, 1), 4, 4, 1, C);
        return ggml_conv_2d_dw_direct(ctx, k4, zs, 1, 1, 0, 0, 1, 1);
    }
    // bilinear zeros-padding grid_sample of feat_t [C, r*r] at grid (gx,gy)
    ggml_tensor *grid_sample(ggml_tensor *feat_t, ggml_tensor *gx, ggml_tensor *gy, int r) {
        const float half = (float) r / 2.0f;
        ggml_tensor *fx = ggml_scale_bias(ctx, gx, half, half - 0.5f);  // [r,r]
        ggml_tensor *fy = ggml_scale_bias(ctx, gy, half, half - 0.5f);
        ggml_tensor *x0 = ggml_floor(ctx, fx), *y0 = ggml_floor(ctx, fy);
        ggml_tensor *wx = ggml_sub(ctx, fx, x0), *wy = ggml_sub(ctx, fy, y0);
        ggml_tensor *one_m_wx = ggml_scale_bias(ctx, wx, -1.0f, 1.0f);
        ggml_tensor *one_m_wy = ggml_scale_bias(ctx, wy, -1.0f, 1.0f);
        ggml_tensor *x1 = ggml_scale_bias(ctx, x0, 1.0f, 1.0f);
        ggml_tensor *y1 = ggml_scale_bias(ctx, y0, 1.0f, 1.0f);

        auto corner = [&](ggml_tensor *cx, ggml_tensor *cy, ggml_tensor *ww) {
            // in-bounds mask (coords are exact integers)
            ggml_tensor *ib = ggml_mul(
                ctx, ggml_clamp(ctx, ggml_scale_bias(ctx, cx, 1.0f, 1.0f), 0.0f, 1.0f),
                ggml_clamp(ctx, ggml_scale_bias(ctx, cx, -1.0f, (float) r), 0.0f, 1.0f));
            ib = ggml_mul(ctx, ib,
                          ggml_mul(ctx, ggml_clamp(ctx, ggml_scale_bias(ctx, cy, 1.0f, 1.0f), 0.0f, 1.0f),
                                   ggml_clamp(ctx, ggml_scale_bias(ctx, cy, -1.0f, (float) r), 0.0f, 1.0f)));
            ggml_tensor *xc = ggml_clamp(ctx, cx, 0.0f, (float) (r - 1));
            ggml_tensor *yc = ggml_clamp(ctx, cy, 0.0f, (float) (r - 1));
            ggml_tensor *idx = ggml_add(ctx, xc, ggml_scale(ctx, yc, (float) r));
            idx = ggml_cast(ctx, ggml_reshape_1d(ctx, idx, (int64_t) r * r), GGML_TYPE_I32);
            ggml_tensor *g = ggml_get_rows(ctx, feat_t, idx);  // [C, r*r]
            ggml_tensor *wgt = ggml_mul(ctx, ww, ib);          // [r,r]
            wgt = ggml_reshape_2d(ctx, wgt, 1, (int64_t) r * r);
            return ggml_mul(ctx, g, wgt);
        };
        ggml_tensor *acc = corner(x0, y0, ggml_mul(ctx, one_m_wx, one_m_wy));
        acc = ggml_add(ctx, acc, corner(x1, y0, ggml_mul(ctx, wx, one_m_wy)));
        acc = ggml_add(ctx, acc, corner(x0, y1, ggml_mul(ctx, one_m_wx, wy)));
        acc = ggml_add(ctx, acc, corner(x1, y1, ggml_mul(ctx, wx, wy)));
        // [C, r*r] -> [r, r, C]
        acc = ggml_reshape_3d(ctx, acc, acc->ne[0], r, r);
        return ggml_cont(ctx, ggml_permute(ctx, acc, 2, 0, 1, 3));
    }
};

bool motion_dec::build(backend_t &be, bool with_taps, std::string &err) {
    frame.release();
    plan_t &p = frame;
    p.begin(4096);
    ggml_context *ctx = p.ctx;
    mdec_build B{p, ctx, w, derived};

    B.latent = p.input("style", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1));
    B.one = p.input("one", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1));

    // constant inputs: zero-stuff masks and identity grids per resolution
    ggml_tensor *masks[8];  // masks[j]: [8<<j, 8<<j] checker (even coords = 1)
    ggml_tensor *xs_x[7], *xs_y[7];
    for (int j = 0; j < 7; j++) {
        const int r = 8 << j;
        char nm[24];
        snprintf(nm, sizeof nm, "mask%d", j);
        masks[j] = p.input(nm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r));
        snprintf(nm, sizeof nm, "xsx%d", j);
        xs_x[j] = p.input(nm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r));
        snprintf(nm, sizeof nm, "xsy%d", j);
        xs_y[j] = p.input(nm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r));
    }
    ggml_tensor *feats[7];
    for (int j = 0; j < 7; j++) {
        const int r = 8 << j;
        char nm[24];
        snprintf(nm, sizeof nm, "feat%d", j);
        // channels of feat j == ST_OC[j] (conv{6-j} channels match stage OC)
        feats[j] = p.input(nm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ST_OC[j], (int64_t) r * r));
    }

    std::vector<tap_entry> taps;
    auto reg = [&](const std::string &name, ggml_tensor *t, std::vector<int64_t> sh) {
        if (!with_taps) return;
        char on[48];
        snprintf(on, sizeof on, "tapout.%zu", taps.size());
        p.output(on, t);
        taps.push_back({name, t, std::move(sh)});
    };

    // constant input + conv1 + (no to_rgb1 in forward)
    ggml_tensor *cinp = B.need("dec.input.input");  // [4,4,512,1]
    if (!cinp) {
        err = B.err;
        return false;
    }
    ggml_tensor *out = B.styled_conv(cinp, "dec.conv1");
    if (!out) {
        err = B.err;
        return false;
    }
    reg("conv1_4", out, {1, 512, 4, 4});

    ggml_tensor *skip = nullptr;      // rgb accumulator
    ggml_tensor *skip_flow = nullptr; // flowmap accumulator (pre-activations)

    for (int j = 0; j < 7; j++) {
        const int r = 8 << j;
        char b1[32], b2[32], bf[32], br[32];
        snprintf(b1, sizeof b1, "dec.convs.%d", 2 * j);
        snprintf(b2, sizeof b2, "dec.convs.%d", 2 * j + 1);
        snprintf(bf, sizeof bf, "dec.to_flows.%d", j);
        snprintf(br, sizeof br, "dec.to_rgbs.%d", j);

        out = B.styled_conv_up(out, b1, masks[j]);
        if (!out) {
            err = B.err;
            return false;
        }
        char tn[32];
        snprintf(tn, sizeof tn, "r%d.conv1", r);
        reg(tn, out, {1, ST_OC[j], r, r});
        out = B.styled_conv(out, b2);
        if (!out) {
            err = B.err;
            return false;
        }
        snprintf(tn, sizeof tn, "r%d.conv2", r);
        reg(tn, out, {1, ST_OC[j], r, r});

        // ---- ToFlow ----
        const std::string fbase = bf;
        ggml_tensor *fstyle = B.style_of(fbase + ".conv");
        ggml_tensor *fW = B.need(fbase + ".conv.weight");
        ggml_tensor *fbias = B.need(fbase + ".bias");
        ggml_tensor *fup = B.need(fbase + ".upsample.kernel");
        if (!fstyle || !fW || !fbias || !fup) {
            err = B.err;
            return false;
        }
        ggml_tensor *xs = ggml_mul(ctx, out, ggml_reshape_4d(ctx, fstyle, 1, 1, fstyle->ne[0], 1));
        ggml_tensor *fl = conv2d(ctx, fW, xs, 1, 0);  // [r,r,3]
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
        if (with_taps) {
            ggml_tensor *grid = ggml_concat(ctx, ggml_reshape_3d(ctx, gx, 1, r, r),
                                            ggml_reshape_3d(ctx, gy, 1, r, r), 0);
            snprintf(tn, sizeof tn, "r%d.grid", r);
            reg(tn, grid, {1, r, r, 2});
        }
        ggml_tensor *m = ggml_sigmoid(ctx, ch2);  // [r,r]
        ggml_tensor *sampled = B.grid_sample(feats[j], gx, gy, r);  // [r,r,C]
        ggml_tensor *m3 = ggml_reshape_3d(ctx, m, r, r, 1);
        ggml_tensor *warp = ggml_mul(ctx, sampled, m3);
        snprintf(tn, sizeof tn, "r%d.warp", r);
        reg(tn, warp, {1, ST_OC[j], r, r});
        ggml_tensor *trunk =
            ggml_add(ctx, warp, ggml_sub(ctx, out, ggml_mul(ctx, out, m3)));
        snprintf(tn, sizeof tn, "r%d.out", r);
        reg(tn, trunk, {1, ST_OC[j], r, r});

        // ---- ToRGB(out_warp, skip) ----
        const std::string rbase = br;
        ggml_tensor *rW = B.need(rbase + ".conv.0.weight");
        ggml_tensor *rlb = B.need(rbase + ".conv.1.bias");
        ggml_tensor *rbias = B.need(rbase + ".bias");
        ggml_tensor *rup = B.need(rbase + ".upsample.kernel");
        if (!rW || !rlb || !rbias || !rup) {
            err = B.err;
            return false;
        }
        ggml_tensor *rgb = conv2d(ctx, rW, warp, 1, 0);
        rgb = fused_lrelu(ctx, rgb, rlb);
        rgb = ggml_add(ctx, rgb, rbias);
        if (skip) rgb = ggml_add(ctx, rgb, B.upsample2x(skip, rup, masks[j]));
        skip = rgb;
        snprintf(tn, sizeof tn, "r%d.rgb", r);
        reg(tn, rgb, {1, 3, r, r});

        out = trunk;
    }

    p.output("img", skip);  // [512,512,3]
    if (!B.err.empty()) {
        err = B.err;
        return false;
    }
    if (!p.commit(be, err)) return false;

    // constant inputs
    const float one = 1.0f;
    p.set("one", &one);
    std::vector<float> buf;
    for (int j = 0; j < 7; j++) {
        const int r = 8 << j;
        buf.assign((size_t) r * r, 0.0f);
        for (int y = 0; y < r; y += 2)
            for (int x = 0; x < r; x += 2) buf[(size_t) y * r + x] = 1.0f;
        char nm[24];
        snprintf(nm, sizeof nm, "mask%d", j);
        p.set(nm, buf.data());
        // identity grid: np.linspace(-1,1,r) meshgrid
        std::vector<double> lin((size_t) r);
        for (int i = 0; i < r; i++) lin[(size_t) i] = -1.0 + 2.0 * i / (r - 1);
        buf.assign((size_t) r * r, 0.0f);
        for (int y = 0; y < r; y++)
            for (int x = 0; x < r; x++) buf[(size_t) y * r + x] = (float) lin[(size_t) x];
        snprintf(nm, sizeof nm, "xsx%d", j);
        p.set(nm, buf.data());
        for (int y = 0; y < r; y++)
            for (int x = 0; x < r; x++) buf[(size_t) y * r + x] = (float) lin[(size_t) y];
        snprintf(nm, sizeof nm, "xsy%d", j);
        p.set(nm, buf.data());
    }

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
        ggml_tensor *t = frame[nm];
        if (!t || (int64_t) m.feats[(size_t) j].size() != ggml_nelements(t)) {
            err = "feat size mismatch";
            return false;
        }
        frame.set(nm, m.feats[(size_t) j].data());
    }
    return true;
}

bool motion_dec::decode(backend_t &be, const float *style, float *img_out,
                        tap_sink &tap, std::string &err) {
    frame.set("style", style);
    if (!frame.compute(be, err)) return false;
    if (tap) {
        for (auto &e : tap_list) tap.pending.push_back(e);
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
    dbuf = nullptr;
    dctx = nullptr;
    derived.clear();
    built = false;
}

}  // namespace tp
