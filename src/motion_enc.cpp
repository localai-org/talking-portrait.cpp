#include "motion_enc.h"

#include "ops_common.h"

#include <cmath>
#include <cstdio>

namespace tp {

// Equal* runtime scales folded at load (see docs/WEIGHTS.md), PLUS constant
// folding of the FusedLeakyReLU sqrt(2) (leaky is positively homogeneous:
// sqrt2*leaky(x+b) == leaky(sqrt2*x + sqrt2*b)) and of the ResBlock 1/sqrt2:
//   (sqrt2*leaky(conv2+b) + skip)/sqrt2 == leaky(conv2 + b) + skip/sqrt2
double motion_ae_scale(const std::string &name, const int64_t *ne) {
    const double SQRT2 = 1.4142135623730951;
    auto ends_with = [&](const char *s) {
        size_t n = strlen(s);
        return name.size() >= n && name.compare(name.size() - n, n, s) == 0;
    };
    // FusedLeakyReLU biases (menc stem/conv1 + ToRGB inner): x sqrt2
    if (name == "enc.net_app.convs.0.1.bias" || ends_with(".conv1.1.bias") ||
        ends_with(".activate.bias") ||
        (name.rfind("dec.to_rgbs.", 0) == 0 && ends_with(".conv.1.bias")))
        return SQRT2;
    if (!ends_with(".weight")) return 1.0;
    if (name.rfind("dec.direction", 0) == 0) return 1.0;
    if (ends_with("noise.weight")) return 1.0;
    if (ends_with("modulation.weight")) return 1.0 / sqrt((double) ne[0]);
    if (name.rfind("enc.fc.", 0) == 0) return 1.0 / sqrt((double) ne[0]);
    const double conv = 1.0 / sqrt((double) (ne[0] * ne[1] * ne[2]));
    // menc convs followed by FLReLU: x sqrt2 (conv2's sqrt2 cancels with the
    // ResBlock 1/sqrt2; skip takes the 1/sqrt2)
    if (name == "enc.net_app.convs.0.0.weight" || ends_with(".conv1.0.weight"))
        return conv * SQRT2;
    if (ends_with(".conv2.1.weight")) return conv;
    if (ends_with(".skip.1.weight")) return conv / SQRT2;
    if (name == "enc.net_app.convs.8.weight") return conv;
    // ToRGB EqualConv2d feeds its FLReLU: x sqrt2
    if (name.rfind("dec.to_rgbs.", 0) == 0 && ends_with(".conv.0.weight"))
        return conv * SQRT2;
    // ModulatedConv2d raw weights (demod convs get their sqrt2 via S_aug/2;
    // ToFlow has no activation)
    if (ends_with(".conv.weight")) return conv;
    return 1.0;
}

static const int RES_IN[7] = {32, 64, 128, 256, 512, 512, 512};
static const int RES_OUT[7] = {64, 128, 256, 512, 512, 512, 512};

bool motion_enc::run(backend_t &be, const float *image, menc_result &out,
                     tap_sink &tap, std::string &err) {
    if (!built) {
        plan_t &p = plan;
        p.begin(1024);
        ggml_context *ctx = p.ctx;
        auto need = [&](const std::string &n) -> ggml_tensor * {
            ggml_tensor *t = w->get(n);
            if (!t) err = "missing tensor " + n;
            return t;
        };
        ggml_tensor *stem_w = need("enc.net_app.convs.0.0.weight");
        ggml_tensor *stem_b = need("enc.net_app.convs.0.1.bias");
        if (!stem_w || !stem_b) return false;
        auto reg = [&](const std::string &name, ggml_tensor *t, std::vector<int64_t> sh) {
            char nm[64];
            snprintf(nm, sizeof nm, "out.%s", name.c_str());
            p.output(nm, t);
            tap_list.push_back({name, t, std::move(sh)});
        };

        ggml_tensor *img = p.input("image", ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 512, 3, 1));

        // stem: EqualConv2d(3,32,1) + FusedLeakyReLU
        ggml_tensor *x = conv2d(ctx, stem_w, img, 1, 0);
        x = fused_lrelu(ctx, x, stem_b);
        reg("menc.conv0", x, {1, 32, 512, 512});

        std::vector<ggml_tensor *> pyramid;  // conv0..conv7 activations
        pyramid.push_back(x);
        int res = 512;
        for (int i = 1; i <= 7; i++) {
            char base[64];
            snprintf(base, sizeof base, "enc.net_app.convs.%d", i);
            const std::string b = base;
            ggml_tensor *w1 = need(b + ".conv1.0.weight");
            ggml_tensor *b1 = need(b + ".conv1.1.bias");
            ggml_tensor *k2 = need(b + ".conv2.0.kernel");
            ggml_tensor *w2 = need(b + ".conv2.1.weight");
            ggml_tensor *b2 = need(b + ".conv2.2.bias");
            ggml_tensor *ks = need(b + ".skip.0.kernel");
            ggml_tensor *ws = need(b + ".skip.1.weight");
            if (!w1 || !b1 || !k2 || !w2 || !b2 || !ks || !ws) return false;
            const int C = RES_IN[i - 1];

            ggml_tensor *x1 = fused_lrelu(ctx, conv2d(ctx, w1, x, 1, 1), b1);
            ggml_tensor *k2c = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, k2, 4, 4, 1, 1), 4, 4, 1, C);
            ggml_tensor *xb = blur(ctx, x1, k2c, 2, 2);
            ggml_tensor *x2 = fused_lrelu(ctx, conv2d(ctx, w2, xb, 2, 0), b2);
            ggml_tensor *ksc = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, ks, 4, 4, 1, 1), 4, 4, 1, C);
            ggml_tensor *sb = blur(ctx, x, ksc, 1, 1);
            ggml_tensor *sk = conv2d(ctx, ws, sb, 2, 0);
            x = ggml_add(ctx, x2, sk);  // 1/sqrt2 folded into weights
            res /= 2;
            char tn[24];
            snprintf(tn, sizeof tn, "menc.conv%d", i);
            reg(tn, x, {1, RES_OUT[i - 1], res, res});
            pyramid.push_back(x);
        }

        // final 4x4 conv (no bias, no activation) -> appearance
        ggml_tensor *c8 = conv2d(ctx, need("enc.net_app.convs.8.weight"), x, 1, 0);
        reg("menc.conv8", c8, {1, 512, 1, 1});
        ggml_tensor *app = ggml_reshape_2d(ctx, c8, 512, 1);
        p.output("appearance", app);

        // fc chain: 5 plain equal-scaled linears
        ggml_tensor *h = app;
        for (int i = 0; i < 5; i++) {
            char nm[32];
            snprintf(nm, sizeof nm, "enc.fc.%d.weight", i);
            char nb[32];
            snprintf(nb, sizeof nb, "enc.fc.%d.bias", i);
            ggml_tensor *fw = need(nm), *fb = need(nb);
            if (!fw || !fb) return false;
            h = linear(ctx, fw, fb, h);
            char tn[16];
            snprintf(tn, sizeof tn, "menc.fc%d", i);
            reg(tn, h, {1, i == 4 ? 20 : 512});
        }
        p.output("lambda", h);

        // motion = Q @ lambda  (Q ggml [20,512])
        ggml_tensor *Q = need("dec.direction.Q");
        if (!Q) return false;
        ggml_tensor *motion = ggml_mul_mat(ctx, Q, h);  // [512,1]
        p.output("motion", motion);

        // decoder warp features, pixel-major [C, r*r]: conv6..conv0
        for (int j = 0; j < 7; j++) {
            ggml_tensor *f = pyramid[6 - j];  // conv{6-j}
            ggml_tensor *ft = ggml_cont(ctx, ggml_permute(ctx, f, 1, 2, 0, 3));
            ft = ggml_reshape_2d(ctx, ft, f->ne[2], f->ne[0] * f->ne[1]);
            char nm[16];
            snprintf(nm, sizeof nm, "feat%d", j);
            p.output(nm, ft);
        }

        if (!err.empty()) return false;
        if (!p.commit(be, err)) return false;
        built = true;
    }

    plan_t &p = plan;
    p.set("image", image);
    if (!p.compute(be, err)) return false;

    if (tap) {
        for (auto &e : tap_list) tap.pending.push_back(e);
        if (!tap.harvest()) {
            err = tap.err;
            return false;
        }
    }

    out.appearance.resize(512);
    p.get("appearance", out.appearance.data());
    out.lambda.resize(20);
    p.get("lambda", out.lambda.data());
    out.motion.resize(512);
    p.get("motion", out.motion.data());
    out.feats.resize(7);
    out.feat_ch.resize(7);
    out.feat_res.resize(7);
    for (int j = 0; j < 7; j++) {
        char nm[16];
        snprintf(nm, sizeof nm, "feat%d", j);
        ggml_tensor *t = p[nm];
        out.feat_ch[j] = (int) t->ne[0];
        out.feat_res[j] = (int) (8 << j);
        out.feats[j].resize((size_t) ggml_nelements(t));
        p.get(nm, out.feats[j].data());
    }
    return true;
}

}  // namespace tp
