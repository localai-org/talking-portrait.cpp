// LIA motion encoder: EncoderApp pyramid (EqualConv2d/FusedLeakyReLU/Blur
// ResBlocks) + 5-layer EqualLinear fc chain + Direction (baked Q).
#pragma once

#include "tp_internal.h"

namespace tp {

struct menc_result {
    std::vector<float> appearance;  // s_r [512]
    std::vector<float> lambda;      // [20]
    std::vector<float> motion;      // r_s [512] = Q @ lambda
    // pyramid features for decoder warping, host copies, pixel-major [C, H*W]
    // index j: stage r = 8<<j warps feats[j] (= menc conv{6-j})
    std::vector<std::vector<float>> feats;
    std::vector<int> feat_ch, feat_res;
};

struct motion_enc {
    weights_t *w = nullptr;  // shared motion_ae weights (enc.* + dec.direction.Q)

    plan_t plan;  // static graph, built on first run (member: must be freed
                  // before the backend, never from a static destructor)
    std::vector<tap_entry> tap_list;
    bool built = false;

    bool run(backend_t &be, const float *image, menc_result &out, tap_sink &tap,
             std::string &err);
};

// Equal* runtime scale folding for motion_ae tensors (docs/WEIGHTS.md)
double motion_ae_scale(const std::string &name, const int64_t *ne);

}  // namespace tp
