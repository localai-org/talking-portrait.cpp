#include "fmt.h"

#include <cmath>
#include <cstdio>

namespace tp {

bool fmt_model::load(const std::string &path, backend_t &be, std::string &err) {
    if (!w.load(path, be, "float-fmt", nullptr, err)) return false;
    dim_h = (int) w.num("float.fmt.dim_h", 1024);
    depth = (int) w.num("float.fmt.depth", 8);
    n_heads = (int) w.num("float.fmt.n_heads", 8);
    dim_w = (int) w.num("float.fmt.dim_w", 512);
    dim_a = (int) w.num("float.fmt.dim_a", 512);
    dim_e = (int) w.num("float.fmt.dim_e", 7);
    chunk_frames = (int) w.num("float.fmt.chunk_frames", 50);
    prev_frames = (int) w.num("float.fmt.prev_frames", 10);
    total = (int) w.num("float.fmt.num_total_frames", 60);
    attn_window = (int) w.num("float.fmt.attn_window", 2);
    t_freq_dim = (int) w.num("float.fmt.t_freq_dim", 256);
    nfe = (int) w.num("float.fmt.nfe", 10);
    a_cfg = (float) w.num("float.fmt.a_cfg_scale", 2.0);
    e_cfg = (float) w.num("float.fmt.e_cfg_scale", 1.0);
    r_cfg = (float) w.num("float.fmt.r_cfg_scale", 1.0);
    ln_eps = (float) w.num("float.fmt.ln_eps", 1e-6);
    mlp_ratio = (float) w.num("float.fmt.mlp_ratio", 4.0);

    // banded alignment mask: row i attends cols [max(0,i-w), i+w+1)
    mask_i32.assign((size_t) total * total, 1);
    for (int i = 0; i < total; i++) {
        for (int j = std::max(0, i - attn_window); j < std::min(total, i + attn_window + 1); j++)
            mask_i32[(size_t) i * total + j] = 0;
    }

    // GLIDE sinusoidal freqs: exp(-ln(10000) * k / half), k = 0..half-1
    const int half = t_freq_dim / 2;
    t_freqs.resize((size_t) half);
    for (int k = 0; k < half; k++)
        t_freqs[(size_t) k] = (float) exp(-log(10000.0) * (double) k / (double) half);
    return true;
}

bool fmt_model::build(backend_t &be, std::string &err) {
    plan_t &p = step;
    p.begin(2048);
    ggml_context *ctx = p.ctx;
    auto need = [&](const std::string &n) -> ggml_tensor * {
        ggml_tensor *t = w.get(n);
        if (!t) err = "missing tensor " + n;
        return t;
    };
    const int T = total, H = dim_h, B = 3;
    const int hd = H / n_heads;

    ggml_tensor *x_in = p.input("x", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_w, T));
    ggml_tensor *cond = p.input("cond", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim_w + dim_a + dim_e, T, B));
    ggml_tensor *tf = p.input("t_freq", ggml_new_tensor_1d(ctx, GGML_TYPE_F32, t_freq_dim));
    ggml_tensor *mask = p.input("mask", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T));

    // t embedding MLP
    ggml_tensor *t_emb = linear(ctx, need("t_embedder.mlp.0.weight"), need("t_embedder.mlp.0.bias"), tf);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = linear(ctx, need("t_embedder.mlp.2.weight"), need("t_embedder.mlp.2.bias"), t_emb);
    p.output("tap.t_embed", t_emb);

    // condition embedding + t
    ggml_tensor *c_lin = linear(ctx, need("c_embedder.weight"), need("c_embedder.bias"), cond);
    p.output("tap.c_embed", c_lin);
    ggml_tensor *c = ggml_add(ctx, c_lin, ggml_reshape_3d(ctx, t_emb, H, 1, 1));
    ggml_tensor *silu_c = ggml_silu(ctx, c);

    // x embedding (+ pos)
    ggml_tensor *x1 = linear(ctx, need("x_embedder.proj.weight"), need("x_embedder.proj.bias"), x_in);
    ggml_tensor *x3 = ggml_repeat_4d(ctx, x1, H, T, B, 1);
    p.output("tap.x_embed", x3);
    ggml_tensor *pos = need("pos_embed");  // ggml [1024, 60, 1]
    if (!pos) return false;
    ggml_tensor *x = ggml_add(ctx, x3, ggml_reshape_3d(ctx, pos, H, T, 1));

    for (int i = 0; i < depth; i++) {
        char nm[80];
        auto WT = [&](const char *fmt) {
            snprintf(nm, sizeof nm, fmt, i);
            return need(nm);
        };
        ggml_tensor *mod = linear(ctx, WT("blocks.%d.adaLN_modulation.1.weight"),
                                  WT("blocks.%d.adaLN_modulation.1.bias"), silu_c);  // [6H, T, B]
        char on[32];
        snprintf(on, sizeof on, "tap.b%d.mod", i);
        p.output(on, mod);
        auto part = [&](int k) {
            return ggml_view_3d(ctx, mod, H, T, B, mod->nb[1], mod->nb[2], (size_t) k * H * 4);
        };
        ggml_tensor *shift_msa = part(0), *scale_msa = part(1), *gate_msa = part(2);
        ggml_tensor *shift_mlp = part(3), *scale_mlp = part(4), *gate_mlp = part(5);

        // attention branch
        ggml_tensor *h = ggml_norm(ctx, x, ln_eps);
        h = ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h, scale_msa), h), shift_msa);
        ggml_tensor *qkv = linear(ctx, WT("blocks.%d.attn.qkv.weight"),
                                  WT("blocks.%d.attn.qkv.bias"), h);  // [3H, T, B]
        auto head_split = [&](int k) {
            ggml_tensor *v = ggml_view_3d(ctx, qkv, H, T, B, qkv->nb[1], qkv->nb[2], (size_t) k * H * 4);
            v = ggml_cont(ctx, v);
            v = ggml_reshape_4d(ctx, v, hd, n_heads, T, B);
            return v;  // [hd, heads, T, B]
        };
        ggml_tensor *q4 = head_split(0), *k4 = head_split(1), *v4 = head_split(2);
        ggml_tensor *qp = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));  // [hd, T, heads, B]
        ggml_tensor *kp = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));
        ggml_tensor *kq = ggml_mul_mat(ctx, kp, qp);  // [Tk, Tq, heads, B]
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
        ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));  // [T, hd, heads, B]
        ggml_tensor *kqv = ggml_mul_mat(ctx, vt, kq);                         // [hd, Tq, heads, B]
        ggml_tensor *merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        merged = ggml_reshape_3d(ctx, merged, H, T, B);
        ggml_tensor *attn = linear(ctx, WT("blocks.%d.attn.proj.weight"),
                                   WT("blocks.%d.attn.proj.bias"), merged);
        snprintf(on, sizeof on, "tap.b%d.attn", i);
        p.output(on, attn);
        x = ggml_add(ctx, x, ggml_mul(ctx, gate_msa, attn));

        // MLP branch (tanh-GELU)
        ggml_tensor *h2 = ggml_norm(ctx, x, ln_eps);
        h2 = ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h2, scale_mlp), h2), shift_mlp);
        ggml_tensor *m1 = linear(ctx, WT("blocks.%d.mlp.fc1.weight"), WT("blocks.%d.mlp.fc1.bias"), h2);
        m1 = ggml_gelu(ctx, m1);
        ggml_tensor *m2 = linear(ctx, WT("blocks.%d.mlp.fc2.weight"), WT("blocks.%d.mlp.fc2.bias"), m1);
        snprintf(on, sizeof on, "tap.b%d.mlp", i);
        p.output(on, m2);
        x = ggml_add(ctx, x, ggml_mul(ctx, gate_mlp, m2));
        snprintf(on, sizeof on, "tap.b%d.out", i);
        p.output(on, x);
    }

    // final decoder head
    ggml_tensor *fmod = linear(ctx, need("decoder.adaLN_modulation.1.weight"),
                               need("decoder.adaLN_modulation.1.bias"), silu_c);  // [2H, T, B]
    p.output("tap.final_mod", fmod);
    ggml_tensor *shift = ggml_view_3d(ctx, fmod, dim_h, T, B, fmod->nb[1], fmod->nb[2], 0);
    ggml_tensor *scale = ggml_view_3d(ctx, fmod, dim_h, T, B, fmod->nb[1], fmod->nb[2], (size_t) dim_h * 4);
    ggml_tensor *h = ggml_norm(ctx, x, ln_eps);
    h = ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, h, scale), h), shift);
    ggml_tensor *v = linear(ctx, need("decoder.linear.weight"), need("decoder.linear.bias"), h);
    p.output("v", v);  // [512, 60, 3]

    if (!err.empty()) return false;
    if (!p.commit(be, err)) return false;

    // additive banded mask (0 = attend, -inf = masked); NOTE: re-set every
    // forward — gallocr may reuse graph-input memory mid-compute
    mask_f32.resize((size_t) T * T);
    for (size_t i = 0; i < mask_f32.size(); i++)
        mask_f32[i] = mask_i32[i] ? -INFINITY : 0.0f;
    return true;
}

bool fmt_model::forward(backend_t &be, const float *x60, const float *cond, float t,
                        float *v_out, tap_sink &tap, std::string &err) {
    plan_t &p = step;
    p.set("x", x60);
    p.set("cond", cond);
    p.set("mask", mask_f32.data());

    const int half = t_freq_dim / 2;
    std::vector<float> tf((size_t) t_freq_dim);
    for (int k = 0; k < half; k++) {
        const float a = t * t_freqs[(size_t) k];
        tf[(size_t) k] = cosf(a);
        tf[(size_t) (half + k)] = sinf(a);
    }
    p.set("t_freq", tf.data());

    if (!p.compute(be, err)) return false;

    if (tap) {
        auto add = [&](const std::string &tapname, const char *out, std::vector<int64_t> sh) {
            tap(tapname, p[out], std::move(sh));
        };
        add("t_embed", "tap.t_embed", {1, dim_h});
        add("c_embed", "tap.c_embed", {3, total, dim_h});
        add("x_embed", "tap.x_embed", {3, total, dim_h});
        for (int i = 0; i < depth; i++) {
            char tn[32], on[32];
            snprintf(tn, sizeof tn, "b%d.mod", i);
            snprintf(on, sizeof on, "tap.b%d.mod", i);
            add(tn, on, {3, total, 6 * dim_h});
            snprintf(tn, sizeof tn, "b%d.attn", i);
            snprintf(on, sizeof on, "tap.b%d.attn", i);
            add(tn, on, {3, total, dim_h});
            snprintf(tn, sizeof tn, "b%d.mlp", i);
            snprintf(on, sizeof on, "tap.b%d.mlp", i);
            add(tn, on, {3, total, dim_h});
            snprintf(tn, sizeof tn, "b%d.out", i);
            snprintf(on, sizeof on, "tap.b%d.out", i);
            add(tn, on, {3, total, dim_h});
        }
        add("final_mod", "tap.final_mod", {3, total, 2 * dim_h});
        add("out", "v", {3, total, dim_w});
        if (!tap.harvest()) {
            err = tap.err;
            return false;
        }
    }

    ggml_tensor *v = p["v"];
    ggml_backend_tensor_get(v, v_out, 0, ggml_nbytes(v));
    return true;
}

}  // namespace tp
