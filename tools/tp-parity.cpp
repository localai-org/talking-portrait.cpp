// parity-dump runner: reads fixture inputs from the harness dumps dir,
// runs the port, writes taps in the docs/TAPS.md format for bin/gate.
//
//   tp-parity --gguf <dir> --ref <harness>/dumps --out parity-dumps \
//             --case a-short [--families aud,menc,ode,fmt,mdec] \
//             [--device cpu|gpu] [--ftype f32|f16]
//   tp-parity ... --generate --frames-out videos   (M6 frame dump)
#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace tp;

struct case_cfg {
    bool menc = false, aud = false;
    int fmt_chunk = -1;              // chunk whose s0 gets full fmt.* taps
    std::string fmt_prefix;          // "fmt." or "fmt.c1s0."
    std::vector<int> full_frames;    // full mdec taps
    std::vector<int> img_frames;     // style+img only
};

static bool get_cfg(const std::string &cs, case_cfg &c) {
    if (cs == "a-short") {
        c = {true, true, 0, "fmt.", {0, 25}, {0, 1, 25, 49}};
    } else if (cs == "b-multichunk") {
        c = {false, false, 1, "fmt.c1s0.", {}, {49, 50, 51}};
    } else if (cs == "c-tiny") {
        c = {false, true, -1, "", {}, {0, 29}};
    } else if (cs == "d-portrait2") {
        c = {true, false, -1, "", {}, {0, 25}};
    } else {
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    std::string gguf_dir, ref_root, out_root = "parity-dumps", case_name;
    std::string device = "cpu", ftype = "f32", families, frames_out;
    bool generate = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf_dir = next();
        else if (a == "--ref") ref_root = next();
        else if (a == "--out") out_root = next();
        else if (a == "--case") case_name = next();
        else if (a == "--families") families = next();
        else if (a == "--device") device = next();
        else if (a == "--ftype") ftype = next();
        else if (a == "--generate") generate = true;
        else if (a == "--frames-out") frames_out = next();
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf_dir.empty() || ref_root.empty() || case_name.empty()) {
        fprintf(stderr, "usage: tp-parity --gguf DIR --ref DUMPS --case NAME "
                        "[--out DIR] [--generate --frames-out DIR]\n");
        return 2;
    }
    case_cfg cfg;
    if (!get_cfg(case_name, cfg)) {
        fprintf(stderr, "unknown case %s\n", case_name.c_str());
        return 2;
    }
    // family filter (default: everything the case defines)
    if (!families.empty()) {
        auto has = [&](const char *f) {
            return families.find(f) != std::string::npos;
        };
        if (!has("menc")) cfg.menc = false;
        if (!has("aud")) cfg.aud = false;
        if (!has("fmt")) cfg.fmt_chunk = -1;
        if (!has("mdec")) {
            cfg.full_frames.clear();
            cfg.img_frames.clear();
        }
        // "ode" family controls c*.* dumps below
    }
    const bool want_ode = families.empty() || families.find("ode") != std::string::npos;

    const std::string case_dir = ref_root + "/" + case_name;
    std::string err;
    fixture fx;
    if (!fx.load(case_dir, err)) {
        fprintf(stderr, "fixture: %s\n", err.c_str());
        return 1;
    }

    tp_ctx *c = tp_load(gguf_dir.c_str(), device.c_str(), 0, ftype.c_str());
    if (!c) return 1;

    // ---------------- M6 generation mode ----------------
    if (generate) {
        if (frames_out.empty()) {
            fprintf(stderr, "--generate needs --frames-out\n");
            return 2;
        }
        std::vector<float> frames((size_t) fx.n_frames * 3 * 512 * 512);
        const int n = tp_generate(c, fx.image.data(), fx.audio.data(), fx.n_samples,
                                  fx.emotion, fx.noise.data(), 0, frames.data());
        if (n < 0) {
            fprintf(stderr, "generate: %s\n", tp_last_error(c));
            return 1;
        }
        const std::string path = frames_out + "/" + case_name + "_frames.f32";
        if (!write_file(path, frames.data(), frames.size() * 4, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        printf("%s: %d frames -> %s\n", case_name.c_str(), n, path.c_str());
        tp_free(c);
        return 0;
    }

    // ---------------- tap dump mode ----------------
    tap_writer wr(out_root + "/" + case_name);
    const int CF = c->fmt.chunk_frames, P = c->fmt.prev_frames, D = c->fmt.dim_w;
    const int64_t n_frames = fx.n_frames, n_chunks = fx.n_chunks;

    // menc (+ feats for the decoder in every case)
    menc_result m;
    {
        tap_sink ts;
        if (cfg.menc) {
            ts.wr = &wr;
            if (!wr.add_f32("menc.inp", fx.image.data(), {1, 3, 512, 512})) {
                fprintf(stderr, "%s\n", wr.err.c_str());
                return 1;
            }
        }
        if (!c->menc.run(c->be, fx.image.data(), m, ts, c->err)) {
            fprintf(stderr, "menc: %s\n", c->err.c_str());
            return 1;
        }
        if (cfg.menc) {
            if (!wr.add_f32("menc.appearance", m.appearance.data(), {1, 512}) ||
                !wr.add_f32("menc.lambda", m.lambda.data(), {1, 20}) ||
                !wr.add_f32("menc.motion", m.motion.data(), {1, 512})) {
                fprintf(stderr, "%s\n", wr.err.c_str());
                return 1;
            }
        }
    }

    // audio
    std::vector<float> wa;
    {
        tap_sink ts;
        if (cfg.aud) ts.wr = &wr;
        if (!c->audio.encode(c->be, fx.audio.data(), fx.n_samples, n_frames, wa, ts, c->err)) {
            fprintf(stderr, "audio: %s\n", c->err.c_str());
            return 1;
        }
    }

    // banded mask (bitwise-gated)
    if (cfg.fmt_chunk >= 0) {
        if (!wr.add_i32("fmt.attn_mask", c->fmt.mask_i32.data(), {60, 60})) {
            fprintf(stderr, "%s\n", wr.err.c_str());
            return 1;
        }
    }

    // ODE loop (honest autoregression), fmt.* taps on chunk cfg.fmt_chunk==0
    std::vector<float> r_d((size_t) n_chunks * CF * D);
    std::vector<float> prev_x((size_t) P * D, 0.0f), prev_wa((size_t) P * D, 0.0f);
    std::vector<float> wa_t((size_t) CF * D);
    for (int64_t ck = 0; ck < n_chunks; ck++) {
        for (int64_t f = 0; f < CF; f++) {
            const int64_t src = std::min(ck * CF + f, n_frames - 1);
            memcpy(wa_t.data() + (size_t) f * D, wa.data() + (size_t) src * D, (size_t) D * 4);
        }
        chunk_io io{};
        io.x0 = fx.noise.data() + (size_t) ck * CF * D;
        io.prev_x = prev_x.data();
        io.prev_wa = prev_wa.data();
        io.cond_audio = wa_t.data();
        io.wr = m.motion.data();
        io.emotion = fx.emotion;
        io.motion_out = r_d.data() + (size_t) ck * CF * D;
        char cp[16];
        snprintf(cp, sizeof cp, "c%lld.", (long long) ck);
        const bool fmt_here = cfg.fmt_chunk == 0 && ck == 0;
        if (!sample_chunk_ex(c, io, want_ode ? &wr : nullptr, cp,
                             fmt_here ? &wr : nullptr, fmt_here ? 0 : -1,
                             "fmt.", c->err)) {
            fprintf(stderr, "ode c%lld: %s\n", (long long) ck, c->err.c_str());
            return 1;
        }
        memcpy(prev_x.data(), io.motion_out + (size_t) (CF - P) * D, (size_t) P * D * 4);
        memcpy(prev_wa.data(), wa_t.data() + (size_t) (CF - P) * D, (size_t) P * D * 4);
    }

    // teacher-forced fmt.c1s0.* forward with REFERENCE prev-context
    if (cfg.fmt_chunk == 1) {
        std::vector<float> rpx, rpw;
        std::vector<int64_t> sh;
        if (!load_ref_tap(case_dir, "c1.prev_x", rpx, sh, err) ||
            !load_ref_tap(case_dir, "c1.prev_wa", rpw, sh, err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        for (int64_t f = 0; f < CF; f++) {
            const int64_t src = std::min(CF + f, n_frames - 1);
            memcpy(wa_t.data() + (size_t) f * D, wa.data() + (size_t) src * D, (size_t) D * 4);
        }
        std::vector<float> mo((size_t) CF * D);
        chunk_io io{};
        io.x0 = fx.noise.data() + (size_t) 1 * CF * D;
        io.prev_x = rpx.data();
        io.prev_wa = rpw.data();
        io.cond_audio = wa_t.data();
        io.wr = m.motion.data();
        io.emotion = fx.emotion;
        io.motion_out = mo.data();
        if (!sample_chunk_ex(c, io, nullptr, "", &wr, 0, cfg.fmt_prefix, c->err)) {
            fprintf(stderr, "fmt.c1s0: %s\n", c->err.c_str());
            return 1;
        }
    }

    // decoder-parity frames: decode REFERENCE c*.motion
    std::vector<int> frames_all = cfg.img_frames;
    for (int f : cfg.full_frames)
        if (std::find(frames_all.begin(), frames_all.end(), f) == frames_all.end())
            frames_all.push_back(f);
    std::sort(frames_all.begin(), frames_all.end());
    if (!frames_all.empty()) {
        std::vector<std::vector<float>> ref_motion((size_t) n_chunks);
        for (int64_t ck = 0; ck < n_chunks; ck++) {
            std::vector<int64_t> sh;
            char nm[24];
            snprintf(nm, sizeof nm, "c%lld.motion", (long long) ck);
            if (!load_ref_tap(case_dir, nm, ref_motion[(size_t) ck], sh, err)) {
                fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
        }
        const bool with_taps = !cfg.full_frames.empty();
        if (!c->mdec.built || c->mdec.taps_in_graph != with_taps) {
            if (!c->mdec.build(c->be, with_taps, c->err)) {
                fprintf(stderr, "mdec build: %s\n", c->err.c_str());
                return 1;
            }
        }
        if (!c->mdec.set_feats(m, c->err)) {
            fprintf(stderr, "%s\n", c->err.c_str());
            return 1;
        }
        std::vector<float> style(512), img((size_t) 3 * 512 * 512);
        for (int f : frames_all) {
            const int ck = f / CF, row = f % CF;
            const float *mo = ref_motion[(size_t) ck].data() + (size_t) row * D;
            for (int i = 0; i < 512; i++) style[(size_t) i] = m.appearance[(size_t) i] + mo[i];
            char pfx[32];
            snprintf(pfx, sizeof pfx, "mdec.f%d.", f);
            tap_sink ts;
            const bool full = std::find(cfg.full_frames.begin(), cfg.full_frames.end(), f) !=
                              cfg.full_frames.end();
            if (full) {
                ts.wr = &wr;
                ts.prefix = pfx;
            }
            if (!c->mdec.decode(c->be, style.data(), img.data(), ts, c->err)) {
                fprintf(stderr, "mdec f%d: %s\n", f, c->err.c_str());
                return 1;
            }
            if (!wr.add_f32(std::string(pfx) + "style", style.data(), {1, 512}) ||
                !wr.add_f32(std::string(pfx) + "img", img.data(), {1, 3, 512, 512})) {
                fprintf(stderr, "%s\n", wr.err.c_str());
                return 1;
            }
        }
    }

    if (!wr.write_meta(case_name)) {
        fprintf(stderr, "meta: %s\n", wr.err.c_str());
        return 1;
    }
    printf("%s: %zu taps -> %s\n", case_name.c_str(), wr.taps.obj.size(),
           (out_root + "/" + case_name).c_str());
    tp_free(c);
    return 0;
}
