// StyleGAN-ish motion decoder (LIA Synthesis): ModulatedConv2d stack with
// ToFlow warping (grid_sample of encoder feats) and ToRGB skip accumulation.
#pragma once

#include "motion_enc.h"
#include "tp_internal.h"

namespace tp {

struct motion_dec {
    weights_t *w = nullptr;  // shared motion_ae weights (dec.*)

    // derived tensors (flipped upsample weights, demod reduction matrices)
    ggml_context *dctx = nullptr;
    ggml_backend_buffer_t dbuf = nullptr;
    std::map<std::string, ggml_tensor *> derived;

    // persistent constants OUTSIDE gallocr (its free-list reuses graph-input
    // memory mid-compute): warp feats, zero-stuff masks, identity grids
    ggml_context *cctx = nullptr;
    ggml_backend_buffer_t cbuf = nullptr;
    std::map<std::string, ggml_tensor *> consts;

    plan_t frame;      // static per-frame graph (built once per process)
    bool built = false;
    bool taps_in_graph = false;  // frame plan variant with tap outputs
    std::vector<tap_entry> tap_list;

    static const int N_STAGES = 7;  // r8..r512

    bool prepare(backend_t &be, std::string &err);  // derived tensors
    // (re)build the per-frame plan; taps=true adds all tap outputs
    bool build(backend_t &be, bool with_taps, std::string &err);
    // set per-portrait features (from menc_result), once per portrait
    bool set_feats(const menc_result &m, std::string &err);
    // style [512] = s_r + r_d[t]  ->  img [3*512*512]
    bool decode(backend_t &be, const float *style, float *img_out, tap_sink &tap,
                std::string &err);

    void release();
    ~motion_dec() { release(); }
};

// weight names whose f32 host copies must be retained at load for prepare()
std::vector<std::string> motion_dec_host_tensors();

}  // namespace tp
