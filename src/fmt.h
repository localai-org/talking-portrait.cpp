// Flow Matching Transformer: 8 DiT blocks with per-frame adaLN modulation,
// banded attention mask, CFG-batched forward [uncond, full, audio_only].
#pragma once

#include "tp_internal.h"

namespace tp {

struct fmt_model {
    weights_t w;
    int dim_h = 1024, depth = 8, n_heads = 8, dim_w = 512, dim_a = 512, dim_e = 7;
    int chunk_frames = 50, prev_frames = 10, total = 60;
    int attn_window = 2, t_freq_dim = 256, nfe = 10;
    float mlp_ratio = 4.0f, ln_eps = 1e-6f;
    float a_cfg = 2.0f, e_cfg = 1.0f, r_cfg = 1.0f;

    plan_t step;  // static CFG-batched forward graph
    std::vector<int32_t> mask_i32;  // [60*60] 1 = masked (tap layout)
    std::vector<float> mask_f32;    // additive mask, re-set every forward

    bool load(const std::string &gguf_path, backend_t &be, std::string &err);
    bool build(backend_t &be, std::string &err);

    // One CFG-batched forward.
    //   x60    : [60*512]   (prev_x rows 0..9, x rows 10..59), shared by rows
    //   cond   : [3*60*1031] host-assembled [wr | wa | we] per branch per frame
    //   t      : flow time in [0,1]
    //   v_out  : [3*60*512] (uncond, full, audio_only)
    // Tap names get prefix from sink; per-block taps registered when sink on.
    bool forward(backend_t &be, const float *x60, const float *cond, float t,
                 float *v_out, tap_sink &tap, std::string &err);

    // host helper: timestep embedding freqs table etc built at load
    std::vector<float> t_freqs;  // [t_freq_dim/2]
    std::vector<float> tf_scratch;  // per-forward t_freq staging (no per-step allocs)
};

}  // namespace tp
