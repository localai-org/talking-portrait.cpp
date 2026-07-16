// audio path: modified wav2vec2-base (group-norm conv stem, temporal interp
// to 25 fps BEFORE the post-norm transformer) + Linear(9216->512)+LN+SiLU.
#pragma once

#include "tp_internal.h"

namespace tp {

struct audio_model {
    weights_t w;
    // hparams (from GGUF KV)
    int n_layers = 12, n_heads = 12, hidden = 768, ffn = 3072, dim_w = 512;
    float ln_eps = 1e-5f;
    static const int N_CONV = 7;
    int conv_dim[N_CONV]{}, conv_kernel[N_CONV]{}, conv_stride[N_CONV]{};
    int pos_kernel = 128, pos_groups = 16;
    int sample_rate = 16000, fps = 25;

    bool load(const std::string &gguf_path, backend_t &be, std::string &err);

    // audio [n_in] (already feature-extractor normalized) -> wa [seq_len*512]
    bool encode(backend_t &be, const float *audio, int64_t n_in, int64_t seq_len,
                std::vector<float> &wa, tap_sink &tap, std::string &err);
};

// f32 conv_1d (ggml_conv_1d casts im2col to f16; parity wants f32)
ggml_tensor *conv_1d_f32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                         int s0, int p0);

}  // namespace tp
