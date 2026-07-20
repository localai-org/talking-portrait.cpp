// the C-API context: backend + the three sub-nets + pipeline state
#pragma once

#include "audio_enc.h"
#include "tp.h"
#include "fmt.h"
#include "motion_dec.h"
#include "motion_enc.h"
#include "tp_internal.h"

struct tp_ctx {
    tp::backend_t be;
    tp::weights_t motion_ae;  // shared by menc + mdec
    tp::audio_model audio;
    tp::motion_enc menc;
    tp::fmt_model fmt;
    tp::motion_dec mdec;
    std::string err;

    bool load(const char *gguf_dir, const char *device, int n_threads, const char *ftype);
};

namespace tp {

// pipeline pieces shared by tp_generate and the parity runner
struct sample_ctx;  // pipeline.cpp

// ODE sampling of one chunk; taps optional (parity). See pipeline.cpp.
struct chunk_io {
    const float *x0;           // [50*512]
    const float *prev_x;       // [10*512]
    const float *prev_wa;      // [10*512]
    const float *cond_audio;   // [50*512]
    const float *wr;           // [512]
    int emotion;               // one-hot index
    float *motion_out;         // [50*512]
};
bool sample_chunk(tp_ctx *c, const chunk_io &io, tap_sink &tap,
                  int fmt_tap_step /*-1=off: full fmt.* taps at this step*/,
                  const std::string &fmt_tap_prefix, std::string &err);
struct tap_writer;
bool sample_chunk_ex(tp_ctx *c, const chunk_io &io, tap_writer *wr,
                     const std::string &chunk_prefix, tap_writer *fmt_wr,
                     int fmt_tap_step, const std::string &fmt_tap_prefix,
                     std::string &err);
// frames_dst: optional contiguous [n_frames][3*512*512] destination written
// directly by the batched decoder (skips the per-frame callback copy)
int generate_impl(tp_ctx *c, const float *image, const float *audio, int64_t n_samples,
                  int emotion, const float *noise, uint64_t seed, tp_frame_cb cb, void *ud,
                  float *frames_dst = nullptr);

}  // namespace tp
