#undef NDEBUG  // asserts are the test
// graph-op invariants vs naive references (no model assets needed):
// upfirdn2d blur path, zero-stuff upsample path, f32 conv1d
#include "audio_enc.h"
#include "ops_common.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tp;

// naive upfirdn2d(up, pad0, pad1) with 4x4 kernel k (per channel identical)
static std::vector<float> naive_upfirdn(const std::vector<float> &x, int W, int C,
                                        const float k[16], int up, int pad0, int pad1) {
    const int UW = W * up;                     // zero-stuffed size (incl trailing zeros)
    const int PW = UW + pad0 + pad1;           // padded
    const int OW = PW - 4 + 1;                 // conv output
    std::vector<float> pad((size_t) C * PW * PW, 0.0f);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < W; y++)
            for (int xx = 0; xx < W; xx++)
                pad[((size_t) c * PW + (y * up + pad0)) * PW + xx * up + pad0] =
                    x[((size_t) c * W + y) * W + xx];
    std::vector<float> out((size_t) C * OW * OW, 0.0f);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < OW; y++)
            for (int xx = 0; xx < OW; xx++) {
                double s = 0;
                for (int a = 0; a < 4; a++)
                    for (int b = 0; b < 4; b++)
                        s += (double) k[a * 4 + b] *
                             pad[((size_t) c * PW + y + a) * PW + xx + b];
                out[((size_t) c * OW + y) * OW + xx] = (float) s;
            }
    return out;
}

int main() {
    std::string err;
    backend_t be;
    assert(be.init("cpu", 2, err));
    std::mt19937 rng(42);
    std::normal_distribution<float> nd;

    const int W = 6, C = 3;
    std::vector<float> x((size_t) C * W * W);
    for (auto &v : x) v = nd(rng);
    float kern[16];
    {
        const float k1[4] = {1, 3, 3, 1};
        float sum = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                kern[i * 4 + j] = k1[i] * k1[j];
                sum += kern[i * 4 + j];
            }
        for (auto &v : kern) v /= sum;
    }

    // ---- blur pad (2,2) ----
    {
        plan_t p;
        p.begin(64);
        ggml_tensor *xi = p.input("x", ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, W, W, C));
        ggml_tensor *ki = p.input("k", ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 4, 4));
        ggml_tensor *kc = ggml_repeat_4d(p.ctx, ggml_reshape_4d(p.ctx, ki, 4, 4, 1, 1), 4, 4, 1, C);
        p.output("y", blur(p.ctx, xi, kc, 2, 2));
        assert(p.commit(be, err));
        p.set("x", x.data());
        p.set("k", kern);
        assert(p.compute(be, err));
        std::vector<float> got((size_t) C * (W + 1) * (W + 1));
        p.get("y", got.data());
        auto ref = naive_upfirdn(x, W, C, kern, 1, 2, 2);
        assert(ref.size() == got.size());
        for (size_t i = 0; i < got.size(); i++) assert(fabs(got[i] - ref[i]) < 1e-5);
    }

    // ---- upsample2x: zero-stuff + pad(2,1) + conv ----
    {
        plan_t p;
        p.begin(64);
        ggml_tensor *xi = p.input("x", ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, W, W, C));
        ggml_tensor *ki = p.input("k", ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 4, 4));
        ggml_tensor *mi = p.input("m", ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 2 * W, 2 * W));
        ggml_tensor *zs = zero_stuff2(p.ctx, xi, mi);
        zs = ggml_pad_ext(p.ctx, zs, 2, 1, 2, 1, 0, 0, 0, 0);
        ggml_tensor *kc = ggml_repeat_4d(p.ctx, ggml_reshape_4d(p.ctx, ki, 4, 4, 1, 1), 4, 4, 1, C);
        p.output("y", ggml_conv_2d_dw_direct(p.ctx, kc, zs, 1, 1, 0, 0, 1, 1));
        assert(p.commit(be, err));
        p.set("x", x.data());
        p.set("k", kern);
        std::vector<float> mask((size_t) 4 * W * W, 0.0f);
        for (int y = 0; y < 2 * W; y += 2)
            for (int xx = 0; xx < 2 * W; xx += 2) mask[(size_t) y * 2 * W + xx] = 1.0f;
        p.set("m", mask.data());
        assert(p.compute(be, err));
        std::vector<float> got((size_t) C * 2 * W * 2 * W);
        p.get("y", got.data());
        auto ref = naive_upfirdn(x, W, C, kern, 2, 2, 1);
        assert(ref.size() == got.size());
        for (size_t i = 0; i < got.size(); i++) assert(fabs(got[i] - ref[i]) < 1e-5);
    }

    // ---- conv_1d_f32 vs naive ----
    {
        const int T = 20, IC = 3, OC = 2, K = 5, S = 2;
        std::vector<float> in((size_t) IC * T), kw((size_t) OC * IC * K);
        for (auto &v : in) v = nd(rng);
        for (auto &v : kw) v = nd(rng);
        plan_t p;
        p.begin(64);
        ggml_tensor *bi = p.input("b", ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, T, IC, 1));
        ggml_tensor *ai = p.input("a", ggml_new_tensor_3d(p.ctx, GGML_TYPE_F32, K, IC, OC));
        p.output("y", conv_1d_f32(p.ctx, ai, bi, S, 0));
        assert(p.commit(be, err));
        p.set("b", in.data());
        p.set("a", kw.data());
        assert(p.compute(be, err));
        const int OT = (T - K) / S + 1;
        std::vector<float> got((size_t) OC * OT);
        p.get("y", got.data());
        for (int o = 0; o < OC; o++)
            for (int t = 0; t < OT; t++) {
                double s = 0;
                for (int c = 0; c < IC; c++)
                    for (int k = 0; k < K; k++)
                        s += (double) kw[((size_t) o * IC + c) * K + k] * in[(size_t) c * T + t * S + k];
                assert(fabs(got[(size_t) o * OT + t] - s) < 1e-4);
            }
    }

    printf("test_ops OK\n");
    return 0;
}
