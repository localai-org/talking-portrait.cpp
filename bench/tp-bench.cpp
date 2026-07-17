// per-stage + pipeline benchmark; emits the bench_torch.py --json row shape:
//   {"label": ..., "rows": [[name, "X ms", "rate"], ...], "peak_vram_mb": N}
//
//   tp-bench --gguf DIR --ref <harness>/dumps [--device gpu] [--ftype f32]
//            [--json bench.json] [--iters 10]
#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tp;

static double now_ms() { return (double) ggml_time_us() / 1000.0; }

int main(int argc, char **argv) {
    std::string gguf, ref_root, device = "gpu", ftype = "f32", json_out;
    // --rows csv over {menc,aud,fmt,ode,mdec,pipe2,pipe10}: skip the others
    // (lets slow fallback configs measure a subset without waiting on the
    // multi-minute pipeline rows). Default: everything.
    std::string rows_sel = "menc,aud,fmt,ode,mdec,pipe2,pipe10";
    int iters = 10;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf = next();
        else if (a == "--ref") ref_root = next();
        else if (a == "--device") device = next();
        else if (a == "--ftype") ftype = next();
        else if (a == "--json") json_out = next();
        else if (a == "--iters") iters = atoi(next());
        else if (a == "--rows") rows_sel = next();
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    auto want = [&](const char *k) {
        return rows_sel.find(k) != std::string::npos;
    };
    if (gguf.empty() || ref_root.empty()) {
        fprintf(stderr, "usage: tp-bench --gguf DIR --ref DUMPS [--json OUT]\n");
        return 2;
    }
    fixture fx;
    std::string err;
    if (!fx.load(ref_root + "/a-short", err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    tp_ctx *c = tp_load(gguf.c_str(), device.c_str(), 0, ftype.c_str());
    if (!c) return 1;

    std::vector<std::vector<std::string>> rows;
    char buf[128];
    tap_sink no_tap;

    // menc
    menc_result m;
    if (!c->menc.run(c->be, fx.image.data(), m, no_tap, c->err)) {
        fprintf(stderr, "menc: %s\n", c->err.c_str());
        return 1;
    }
    if (want("menc")) {
        double t0 = now_ms();
        for (int i = 0; i < iters; i++)
            c->menc.run(c->be, fx.image.data(), m, no_tap, c->err);
        double ms = (now_ms() - t0) / iters;
        snprintf(buf, sizeof buf, "%.1f ms", ms);
        rows.push_back({"menc (per portrait)", buf, ""});
    }

    // audio enc per 10 s (160000 samples, 250 frames)
    if (want("aud")) {
        std::vector<float> a10((size_t) 160000);
        for (size_t i = 0; i < a10.size(); i++) a10[i] = fx.audio[i % fx.audio.size()];
        std::vector<float> wa;
        c->audio.encode(c->be, a10.data(), 160000, 250, wa, no_tap, c->err);  // warmup
        double t0 = now_ms();
        for (int i = 0; i < iters; i++)
            c->audio.encode(c->be, a10.data(), 160000, 250, wa, no_tap, c->err);
        double ms = (now_ms() - t0) / iters;
        snprintf(buf, sizeof buf, "%.1f ms", ms);
        char rate[64];
        snprintf(rate, sizeof rate, "%.0f s-audio/s", 10.0 / (ms / 1000.0));
        rows.push_back({"audio enc (per 10 s)", buf, rate});
    }

    // FMT forward + ODE chunk
    if (want("fmt") || want("ode")) {
        std::vector<float> wa;
        c->audio.encode(c->be, fx.audio.data(), fx.n_samples, fx.n_frames, wa, no_tap, c->err);
        std::vector<float> x60((size_t) 60 * 512, 0.0f), cond((size_t) 3 * 60 * 1031, 0.0f);
        std::vector<float> v((size_t) 3 * 60 * 512);
        c->fmt.forward(c->be, x60.data(), cond.data(), 0.0f, v.data(), no_tap, c->err);
        if (want("fmt")) {
            double t0 = now_ms();
            for (int i = 0; i < iters; i++)
                c->fmt.forward(c->be, x60.data(), cond.data(), 0.5f, v.data(), no_tap, c->err);
            double ms = (now_ms() - t0) / iters;
            snprintf(buf, sizeof buf, "%.2f ms", ms);
            rows.push_back({"FMT fwd (CFG batch 3)", buf, ""});
        }

        std::vector<float> mo((size_t) 50 * 512);
        std::vector<float> zeros((size_t) 10 * 512, 0.0f);
        chunk_io io{};
        io.x0 = fx.noise.data();
        io.prev_x = zeros.data();
        io.prev_wa = zeros.data();
        io.cond_audio = wa.data();
        io.wr = m.motion.data();
        io.emotion = fx.emotion;
        io.motion_out = mo.data();
        if (want("ode")) {
            sample_chunk_ex(c, io, nullptr, "", nullptr, -1, "", c->err);  // warmup
            double t1 = now_ms();
            for (int i = 0; i < iters; i++)
                sample_chunk_ex(c, io, nullptr, "", nullptr, -1, "", c->err);
            double ms1 = (now_ms() - t1) / iters;
            snprintf(buf, sizeof buf, "%.1f ms", ms1);
            char rate[64];
            snprintf(rate, sizeof rate, "%.2f ms/step", ms1 / 9.0);
            rows.push_back({"ODE chunk (9 steps, 50 fr)", buf, rate});
        }
    }

    // mdec per frame
    if (want("mdec")) {
        if (!c->mdec.built || c->mdec.taps_in_graph) c->mdec.build(c->be, false, c->err);
        c->mdec.set_feats(m, c->err);
        std::vector<float> style(512), img((size_t) 3 * 512 * 512);
        for (int i = 0; i < 512; i++) style[(size_t) i] = m.appearance[(size_t) i];
        c->mdec.decode(c->be, style.data(), img.data(), no_tap, c->err);  // warmup x2
        c->mdec.decode(c->be, style.data(), img.data(), no_tap, c->err);
        double t0 = now_ms();
        const int it = iters * 5;
        for (int i = 0; i < it; i++)
            c->mdec.decode(c->be, style.data(), img.data(), no_tap, c->err);
        double ms = (now_ms() - t0) / it;
        snprintf(buf, sizeof buf, "%.2f ms", ms);
        char rate[64];
        snprintf(rate, sizeof rate, "%.1f fps", 1000.0 / ms);
        rows.push_back({"mdec (per frame)", buf, rate});
    }

    // pipeline 2 s and 10 s
    for (double secs : {2.0, 10.0}) {
        if (!(secs == 2.0 ? want("pipe2") : want("pipe10"))) continue;
        const int64_t n = (int64_t) (secs * 16000);
        std::vector<float> audio((size_t) n);
        for (int64_t i = 0; i < n; i++) audio[(size_t) i] = fx.audio[(size_t) (i % fx.audio.size())];
        const int64_t nf = tp_n_frames(n);
        std::vector<float> frames((size_t) nf * 3 * 512 * 512);
        tp_generate(c, fx.image.data(), audio.data(), n, fx.emotion, nullptr, 15,
                    frames.data());  // warmup
        double t0 = now_ms();
        tp_generate(c, fx.image.data(), audio.data(), n, fx.emotion, nullptr, 15,
                    frames.data());
        double ms = now_ms() - t0;
        snprintf(buf, sizeof buf, "%.0f ms", ms);
        char rate[64];
        snprintf(rate, sizeof rate, "RTF %.2f, %.1f fps", secs / (ms / 1000.0),
                 (double) nf / (ms / 1000.0));
        char name[64];
        snprintf(name, sizeof name, "pipeline %g s video", secs);
        rows.push_back({name, buf, rate});
    }

    // report
    std::string js = "{\n \"label\": \"cpp-" + ftype + "-" + device + "\",\n \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); i++) {
        printf("| %-28s | %10s | %20s |\n", rows[i][0].c_str(), rows[i][1].c_str(),
               rows[i][2].c_str());
        js += "  [\"" + rows[i][0] + "\", \"" + rows[i][1] + "\", \"" + rows[i][2] + "\"]";
        js += i + 1 < rows.size() ? ",\n" : "\n";
    }
    size_t vfree = 0, vtotal = 0;
    if (ggml_backend_dev_t dev = ggml_backend_get_device(c->be.be))
        ggml_backend_dev_memory(dev, &vfree, &vtotal);
    char vb[64];
    snprintf(vb, sizeof vb, " ],\n \"peak_vram_mb\": %zu\n}\n",
             vtotal && vfree ? (vtotal - vfree) >> 20 : 0);
    js += vb;
    if (!json_out.empty()) {
        if (!write_file(json_out, js.data(), js.size(), err)) {
            fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    }
    tp_free(c);
    return 0;
}
