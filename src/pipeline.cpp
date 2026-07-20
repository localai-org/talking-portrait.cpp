#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace tp {

// torch linspace(0,1,10) in f32
static inline float ode_t(int i, int nfe) { return (float) ((double) i / (double) (nfe - 1)); }

bool sample_chunk_ex(tp_ctx *c, const chunk_io &io, tap_writer *wr,
                     const std::string &chunk_prefix, tap_writer *fmt_wr,
                     int fmt_tap_step, const std::string &fmt_tap_prefix,
                     std::string &err) {
    fmt_model &F = c->fmt;
    const int T = F.total, P = F.prev_frames, CF = F.chunk_frames;
    const int D = F.dim_w, CD = F.dim_w + F.dim_a + F.dim_e;

    // condition rows: [wr | wa60 | we]; uncond nulls wa (current 50 only) +
    // we; audio-only nulls we; prev_wa rows stay real in ALL branches
    std::vector<float> cond((size_t) 3 * T * CD, 0.0f);
    for (int b = 0; b < 3; b++) {
        for (int f = 0; f < T; f++) {
            float *row = cond.data() + ((size_t) b * T + f) * CD;
            memcpy(row, io.wr, (size_t) D * 4);
            const float *wa_f = f < P ? io.prev_wa + (size_t) f * D
                                      : io.cond_audio + (size_t) (f - P) * D;
            if (!(b == 0 && f >= P)) memcpy(row + D, wa_f, (size_t) D * 4);
            if (b == 1 && io.emotion >= 0 && io.emotion < F.dim_e)
                row[D + F.dim_a + io.emotion] = 1.0f;
        }
    }

    std::vector<float> x((size_t) CF * D);
    memcpy(x.data(), io.x0, x.size() * 4);
    std::vector<float> x60((size_t) T * D);
    memcpy(x60.data(), io.prev_x, (size_t) P * D * 4);
    std::vector<float> v((size_t) 3 * T * D), vc((size_t) T * D);

    auto dump = [&](const std::string &name, const float *data, std::vector<int64_t> sh) {
        if (wr && !wr->add_f32(chunk_prefix + name, data, sh)) {
            err = wr->err;
            return false;
        }
        return true;
    };
    if (wr) {
        if (!dump("x0", io.x0, {1, CF, D})) return false;
        if (!dump("prev_x", io.prev_x, {1, P, D})) return false;
        if (!dump("prev_wa", io.prev_wa, {1, P, D})) return false;
        if (!dump("cond_audio", io.cond_audio, {1, CF, D})) return false;
    }

    for (int s = 0; s < F.nfe - 1; s++) {
        memcpy(x60.data() + (size_t) P * D, x.data(), x.size() * 4);
        tap_sink fs;
        if (fmt_wr && s == fmt_tap_step) {
            fs.wr = fmt_wr;
            fs.prefix = fmt_tap_prefix;
        }
        if (!F.forward(c->be, x60.data(), cond.data(), ode_t(s, F.nfe), v.data(), fs, err))
            return false;

        const float *u = v.data(), *fu = v.data() + (size_t) T * D,
                    *a = v.data() + (size_t) 2 * T * D;
        for (size_t i = 0; i < (size_t) T * D; i++)
            vc[i] = (u[i] + F.a_cfg * (a[i] - u[i])) + F.e_cfg * (fu[i] - a[i]);

        char sp[24];
        snprintf(sp, sizeof sp, "s%d.", s);
        if (wr) {
            if (!dump(std::string(sp) + "v_uncond", u, {1, T, D})) return false;
            if (!dump(std::string(sp) + "v_full", fu, {1, T, D})) return false;
            if (!dump(std::string(sp) + "v_audio", a, {1, T, D})) return false;
            if (!dump(std::string(sp) + "v_cfg", vc.data(), {1, T, D})) return false;
        }

        const float dt = ode_t(s + 1, F.nfe) - ode_t(s, F.nfe);
        const float *vcur = vc.data() + (size_t) P * D;  // v[:, prev:]
        for (size_t i = 0; i < x.size(); i++) x[i] = x[i] + dt * vcur[i];
        if (wr && !dump(std::string(sp) + "x", x.data(), {1, CF, D})) return false;
    }

    memcpy(io.motion_out, x.data(), x.size() * 4);
    if (wr && !dump("motion", x.data(), {1, CF, D})) return false;
    return true;
}

bool sample_chunk(tp_ctx *c, const chunk_io &io, tap_sink &tap, int fmt_tap_step,
                  const std::string &fmt_tap_prefix, std::string &err) {
    return sample_chunk_ex(c, io, tap.wr, tap.prefix, tap.wr, fmt_tap_step,
                           fmt_tap_prefix, err);
}

// ---------------- end-to-end ----------------

int generate_impl(tp_ctx *c, const float *image, const float *audio, int64_t n_samples,
                  int emotion, const float *noise, uint64_t seed, tp_frame_cb cb, void *ud) {
    std::string &err = c->err;
    const int64_t n_frames = tp_n_frames(n_samples);
    const int64_t n_chunks = tp_n_chunks(n_frames);
    if (n_frames <= 0) {
        err = "audio too short";
        return -1;
    }
    const int CF = c->fmt.chunk_frames, P = c->fmt.prev_frames, D = c->fmt.dim_w;

    menc_result m;
    tap_sink no_tap;
    if (!c->menc.run(c->be, image, m, no_tap, err)) return -1;

    std::vector<float> wa;
    if (!c->audio.encode(c->be, audio, n_samples, n_frames, wa, no_tap, err)) return -1;

    std::vector<float> r_d((size_t) n_chunks * CF * D);
    std::vector<float> prev_x((size_t) P * D, 0.0f), prev_wa((size_t) P * D, 0.0f);
    std::vector<float> wa_t((size_t) CF * D);
    std::vector<float> own_noise;
    if (!noise) {
        // upstream fix_noise_seed: fresh generator, SAME seed every chunk
        own_noise.resize((size_t) CF * D);
        std::mt19937_64 g(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto &f : own_noise) f = nd(g);
    }

    for (int64_t ck = 0; ck < n_chunks; ck++) {
        // wa slice, replicate-padded in feature space
        const int64_t f0 = ck * CF;
        for (int64_t f = 0; f < CF; f++) {
            const int64_t src = std::min(f0 + f, n_frames - 1);
            memcpy(wa_t.data() + (size_t) f * D, wa.data() + (size_t) src * D, (size_t) D * 4);
        }
        chunk_io io{};
        io.x0 = noise ? noise + (size_t) ck * CF * D : own_noise.data();
        io.prev_x = prev_x.data();
        io.prev_wa = prev_wa.data();
        io.cond_audio = wa_t.data();
        io.wr = m.motion.data();
        io.emotion = emotion;
        io.motion_out = r_d.data() + (size_t) ck * CF * D;
        if (!sample_chunk_ex(c, io, nullptr, "", nullptr, -1, "", err)) return -1;
        // AR context for the next chunk
        memcpy(prev_x.data(), io.motion_out + (size_t) (CF - P) * D, (size_t) P * D * 4);
        memcpy(prev_wa.data(), wa_t.data() + (size_t) (CF - P) * D, (size_t) P * D * 4);
    }

    if (!c->mdec.built || c->mdec.taps_in_graph) {
        if (!c->mdec.build(c->be, false, err)) return -1;
    }
    if (!c->mdec.set_feats(m, err)) return -1;

    // batched decode: BQ frames per graph dispatch (last batch padded by
    // repeating the final frame's style; pad frames are discarded)
    const int BQ = c->mdec.batch;
    const size_t fsz = (size_t) 3 * 512 * 512;
    std::vector<float> styles((size_t) BQ * 512), frames((size_t) BQ * fsz);
    for (int64_t t0 = 0; t0 < n_frames; t0 += BQ) {
        for (int64_t b = 0; b < BQ; b++) {
            const int64_t t = std::min(t0 + b, n_frames - 1);
            float *st = styles.data() + (size_t) b * 512;
            for (int i = 0; i < 512; i++)
                st[(size_t) i] = m.appearance[(size_t) i] + r_d[(size_t) t * D + i];
        }
        if (!c->mdec.decode_batch(c->be, styles.data(), frames.data(), err)) return -1;
        if (cb) {
            const int64_t nb = std::min<int64_t>(BQ, n_frames - t0);
            for (int64_t b = 0; b < nb; b++) {
                const int rc = cb(t0 + b, frames.data() + (size_t) b * fsz, ud);
                if (rc) return (int) n_frames;
            }
        }
    }
    return (int) n_frames;
}

}  // namespace tp
