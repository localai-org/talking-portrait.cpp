#include "audio_enc.h"

#include <cmath>
#include <cstdio>

namespace tp {

ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                         int s0, int p0) {
    // a: [K, IC, OC], b: [T, IC, 1] -> [OT, OC, 1]
    ggml_tensor *im = ggml_im2col(ctx, a, b, s0, 0, p0, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor *r = ggml_mul_mat(
        ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]));
    return ggml_reshape_3d(ctx, r, im->ne[1], a->ne[2], im->ne[2]);
}

bool audio_model::load(const std::string &path, backend_t &be, std::string &err) {
    if (!w.load(path, be, "float-audio", nullptr, err)) return false;
    n_layers = (int) w.num("float.audio.n_layers", 12);
    n_heads = (int) w.num("float.audio.n_heads", 12);
    hidden = (int) w.num("float.audio.hidden_size", 768);
    ffn = (int) w.num("float.audio.ffn_size", 3072);
    dim_w = (int) w.num("float.audio.dim_w", 512);
    ln_eps = (float) w.num("float.audio.ln_eps", 1e-5);
    pos_kernel = (int) w.num("float.audio.num_conv_pos_embeddings", 128);
    pos_groups = (int) w.num("float.audio.num_conv_pos_embedding_groups", 16);
    sample_rate = (int) w.num("float.audio.sample_rate", 16000);
    fps = (int) w.num("float.audio.fps", 25);
    for (int i = 0; i < N_CONV; i++) {
        char key[64];
        snprintf(key, sizeof key, "float.audio.conv_dim.%d", i);
        conv_dim[i] = (int) w.num(key, 512);
        snprintf(key, sizeof key, "float.audio.conv_kernel.%d", i);
        conv_kernel[i] = (int) w.num(key, i == 0 ? 10 : (i < 5 ? 3 : 2));
        snprintf(key, sizeof key, "float.audio.conv_stride.%d", i);
        conv_stride[i] = (int) w.num(key, i == 0 ? 5 : 2);
    }
    return true;
}

bool audio_model::encode(backend_t &be, const float *audio, int64_t n_in,
                         int64_t seq_len, std::vector<float> &wa,
                         tap_sink &tap, std::string &err) {
    // upstream AudioEncoder.inference: replicate-pad to seq_len*640 samples
    // when not already an exact multiple
    const int64_t unit = (int64_t) (seq_len * sample_rate / fps);
    int64_t n = n_in;
    std::vector<float> padded;
    if (unit > 0 && n % unit != 0) {
        n = unit;  // upstream pads (or crops) to exactly seq_len*640
        padded.assign((size_t) n, n_in > 0 ? audio[n_in - 1] : 0.0f);
        memcpy(padded.data(), audio, (size_t) std::min(n, n_in) * 4);
        audio = padded.data();
    }

    // conv stack output length
    int64_t t_fe = n;
    for (int i = 0; i < N_CONV; i++) t_fe = (t_fe - conv_kernel[i]) / conv_stride[i] + 1;
    if (t_fe < 2 || seq_len < 1) {
        err = "audio too short";
        return false;
    }

    // align_corners=True linear interpolation matrix [t_fe, seq_len]
    std::vector<float> M((size_t) (t_fe * seq_len), 0.0f);
    for (int64_t t = 0; t < seq_len; t++) {
        double pos = seq_len == 1 ? 0.0
                                  : (double) t * (double) (t_fe - 1) / (double) (seq_len - 1);
        int64_t i0 = (int64_t) pos;
        if (i0 > t_fe - 1) i0 = t_fe - 1;
        double frac = pos - (double) i0;
        M[(size_t) (t * t_fe + i0)] += (float) (1.0 - frac);
        if (i0 + 1 < t_fe && frac > 0) M[(size_t) (t * t_fe + i0 + 1)] += (float) frac;
    }

    plan_t p;
    p.begin(4096);
    ggml_context *ctx = p.ctx;

    ggml_tensor *x_in = p.input("audio", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, 1, 1));
    ggml_tensor *m_in = p.input("interpM", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_fe, seq_len));

    // ---- feature extractor ----
    ggml_tensor *x = x_in;
    for (int i = 0; i < N_CONV; i++) {
        char nm[80];
        snprintf(nm, sizeof nm, "feature_extractor.conv_layers.%d.conv.weight", i);
        ggml_tensor *k = w.get(nm);
        if (!k) {
            err = std::string("missing tensor ") + nm;
            return false;
        }
        x = conv_1d_f32(ctx, k, x, conv_stride[i], 0);  // [OT, OC, 1]
        if (i == 0) {
            // GroupNorm(512 groups over 512 ch) == per-channel norm over time
            ggml_tensor *gw = w.get("feature_extractor.conv_layers.0.layer_norm.weight");
            ggml_tensor *gb = w.get("feature_extractor.conv_layers.0.layer_norm.bias");
            x = ggml_norm(ctx, x, ln_eps);
            x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, gw, 1, gw->ne[0]));
            x = ggml_add(ctx, x, ggml_reshape_2d(ctx, gb, 1, gb->ne[0]));
        }
        x = ggml_gelu_erf(ctx, x);
        // tap = full first conv layer output (conv + GroupNorm + GELU)
        if (i == 0 && tap)
            tap("aud.w2v.fe.conv0", p.output("tap.fe.conv0", x), {1, conv_dim[0], x->ne[0]});
    }
    if (tap) tap("aud.w2v.fe.out", p.output("tap.fe.out", x), {1, conv_dim[N_CONV - 1], t_fe});

    // ---- temporal interpolation (align_corners=True) to seq_len frames ----
    // x [t_fe, 512] ; M [t_fe, seq_len] -> interp [512, seq_len]
    ggml_tensor *interp = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]), m_in);
    if (tap) tap("aud.interp", p.output("tap.interp", interp), {1, seq_len, 512});

    // ---- feature projection ----
    ggml_tensor *h = layer_norm(ctx, interp, w.get("feature_projection.layer_norm.weight"),
                                w.get("feature_projection.layer_norm.bias"), ln_eps);
    h = linear(ctx, w.get("feature_projection.projection.weight"),
               w.get("feature_projection.projection.bias"), h);  // [768, S]
    if (tap) tap("aud.w2v.embed", p.output("tap.embed", h), {1, seq_len, hidden});

    // ---- positional conv embedding (grouped conv, k=128, g=16, pad 64) ----
    {
        ggml_tensor *pw = w.get("encoder.pos_conv_embed.conv.weight");  // [128,48,768]
        ggml_tensor *pb = w.get("encoder.pos_conv_embed.conv.bias");
        ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, h));  // [S, 768]
        const int cg = hidden / pos_groups;                        // 48
        ggml_tensor *pos = nullptr;
        for (int g = 0; g < pos_groups; g++) {
            ggml_tensor *kg = ggml_cont(
                ctx, ggml_view_3d(ctx, pw, pw->ne[0], pw->ne[1], cg, pw->nb[1], pw->nb[2],
                                  (size_t) g * cg * pw->nb[2]));
            ggml_tensor *xg = ggml_cont(
                ctx, ggml_view_2d(ctx, xt, xt->ne[0], cg, xt->nb[1], (size_t) g * cg * xt->nb[1]));
            ggml_tensor *og = conv_1d_f32(ctx, kg, xg, 1, pos_kernel / 2);  // [S+1, 48]
            pos = pos ? ggml_concat(ctx, pos, og, 1) : og;
        }
        pos = ggml_reshape_2d(ctx, pos, pos->ne[0], hidden);  // [S+1, 768]
        pos = ggml_add(ctx, pos, ggml_reshape_2d(ctx, pb, 1, hidden));
        // remove last time step, GELU, back to [768, S]
        pos = ggml_view_2d(ctx, pos, seq_len, hidden, pos->nb[1], 0);
        pos = ggml_gelu_erf(ctx, ggml_cont(ctx, pos));
        pos = ggml_cont(ctx, ggml_transpose(ctx, pos));  // [768, S]
        h = ggml_add(ctx, h, pos);
    }
    h = layer_norm(ctx, h, w.get("encoder.layer_norm.weight"),
                   w.get("encoder.layer_norm.bias"), ln_eps);

    // ---- transformer (post-norm), collect all 12 layer outputs ----
    const int hd = hidden / n_heads;  // 64
    const float scaling = 1.0f / sqrtf((float) hd);
    ggml_tensor *stacked = nullptr;
    for (int i = 0; i < n_layers; i++) {
        auto W = [&](const char *fmt) {
            char nm[96];
            snprintf(nm, sizeof nm, fmt, i);
            return w.get(nm);
        };
        ggml_tensor *q = linear(ctx, W("encoder.layers.%d.attention.q_proj.weight"),
                                W("encoder.layers.%d.attention.q_proj.bias"), h);
        q = ggml_scale(ctx, q, scaling);
        ggml_tensor *kk = linear(ctx, W("encoder.layers.%d.attention.k_proj.weight"),
                                 W("encoder.layers.%d.attention.k_proj.bias"), h);
        ggml_tensor *v = linear(ctx, W("encoder.layers.%d.attention.v_proj.weight"),
                                W("encoder.layers.%d.attention.v_proj.bias"), h);
        ggml_tensor *qp = ggml_cont(
            ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, n_heads, seq_len), 0, 2, 1, 3));
        ggml_tensor *kp = ggml_cont(
            ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, kk, hd, n_heads, seq_len), 0, 2, 1, 3));
        ggml_tensor *kq = ggml_soft_max(ctx, ggml_mul_mat(ctx, kp, qp));  // [Sk, Sq, H]
        ggml_tensor *vt = ggml_cont(
            ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, n_heads, seq_len), 1, 2, 0, 3));
        ggml_tensor *kqv = ggml_mul_mat(ctx, vt, kq);  // [hd, Sq, H]
        ggml_tensor *merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        merged = ggml_reshape_2d(ctx, merged, hidden, seq_len);
        ggml_tensor *attn_out = linear(ctx, W("encoder.layers.%d.attention.out_proj.weight"),
                                       W("encoder.layers.%d.attention.out_proj.bias"), merged);
        if (tap && (i == 0 || i == n_layers - 1)) {
            char tn[48];
            snprintf(tn, sizeof tn, "aud.w2v.l%d.attn_out", i);
            char on[48];
            snprintf(on, sizeof on, "tap.l%d.attn_out", i);
            tap(tn, p.output(on, attn_out), {1, seq_len, hidden});
        }
        h = ggml_add(ctx, h, attn_out);
        h = layer_norm(ctx, h, W("encoder.layers.%d.layer_norm.weight"),
                       W("encoder.layers.%d.layer_norm.bias"), ln_eps);
        ggml_tensor *ff = linear(ctx, W("encoder.layers.%d.feed_forward.intermediate_dense.weight"),
                                 W("encoder.layers.%d.feed_forward.intermediate_dense.bias"), h);
        ff = ggml_gelu_erf(ctx, ff);
        ff = linear(ctx, W("encoder.layers.%d.feed_forward.output_dense.weight"),
                    W("encoder.layers.%d.feed_forward.output_dense.bias"), ff);
        if (tap && (i == 0 || i == n_layers - 1)) {
            char tn[48];
            snprintf(tn, sizeof tn, "aud.w2v.l%d.ffn_out", i);
            char on[48];
            snprintf(on, sizeof on, "tap.l%d.ffn_out", i);
            tap(tn, p.output(on, ff), {1, seq_len, hidden});
        }
        h = ggml_add(ctx, h, ff);
        h = layer_norm(ctx, h, W("encoder.layers.%d.final_layer_norm.weight"),
                       W("encoder.layers.%d.final_layer_norm.bias"), ln_eps);
        if (tap) {
            char tn[48];
            snprintf(tn, sizeof tn, "aud.w2v.l%d", i);
            char on[48];
            snprintf(on, sizeof on, "tap.l%d", i);
            tap(tn, p.output(on, h), {1, seq_len, hidden});
        }
        stacked = stacked ? ggml_concat(ctx, stacked, h, 0) : h;
    }

    // ---- audio projection: Linear(9216->512) + LayerNorm + SiLU ----
    ggml_tensor *out = linear(ctx, w.get("audio_projection.0.weight"),
                              w.get("audio_projection.0.bias"), stacked);
    out = layer_norm(ctx, out, w.get("audio_projection.1.weight"),
                     w.get("audio_projection.1.bias"), 1e-5f);
    out = ggml_silu(ctx, out);  // [512, S]
    p.output("wa", out);
    if (tap) tap("aud.proj", out, {1, seq_len, dim_w});

    if (!p.commit(be, err)) return false;
    p.set("audio", audio);
    p.set("interpM", M.data());
    if (!p.compute(be, err)) return false;
    if (!tap.harvest()) {
        err = tap.err;
        return false;
    }

    wa.resize((size_t) (seq_len * dim_w));
    p.get("wa", wa.data());
    return true;
}

}  // namespace tp
