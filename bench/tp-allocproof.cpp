// zero-allocation evidence (M7): counting allocator hooks around the
// steady-state frame loop. Overrides global malloc/free/realloc/calloc and
// operator new/delete; after a 2-frame warmup, a >=100-frame decode loop
// must perform 0 heap allocations per frame.
//
//   tp-allocproof --gguf DIR --ref <harness>/dumps [--device gpu]
//                 [--frames 120] [--out alloc-evidence.txt]
#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>

static std::atomic<long> g_allocs{0}, g_frees{0}, g_bytes{0};
static std::atomic<bool> g_count{false};

extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void *);
extern "C" void *__libc_realloc(void *, size_t);
extern "C" void *__libc_calloc(size_t, size_t);

extern "C" void *malloc(size_t n) {
    if (g_count.load(std::memory_order_relaxed)) {
        g_allocs.fetch_add(1, std::memory_order_relaxed);
        g_bytes.fetch_add((long) n, std::memory_order_relaxed);
    }
    return __libc_malloc(n);
}
extern "C" void free(void *p) {
    if (p && g_count.load(std::memory_order_relaxed))
        g_frees.fetch_add(1, std::memory_order_relaxed);
    __libc_free(p);
}
extern "C" void *realloc(void *p, size_t n) {
    if (g_count.load(std::memory_order_relaxed))
        g_allocs.fetch_add(1, std::memory_order_relaxed);
    return __libc_realloc(p, n);
}
extern "C" void *calloc(size_t n, size_t m) {
    if (g_count.load(std::memory_order_relaxed))
        g_allocs.fetch_add(1, std::memory_order_relaxed);
    return __libc_calloc(n, m);
}

using namespace tp;

int main(int argc, char **argv) {
    std::string gguf, ref_root, device = "gpu", out = "alloc-evidence.txt";
    int frames = 120;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf = next();
        else if (a == "--ref") ref_root = next();
        else if (a == "--device") device = next();
        else if (a == "--frames") frames = atoi(next());
        else if (a == "--out") out = next();
    }
    if (gguf.empty() || ref_root.empty()) {
        fprintf(stderr, "usage: tp-allocproof --gguf DIR --ref DUMPS\n");
        return 2;
    }
    fixture fx;
    std::string err;
    if (!fx.load(ref_root + "/a-short", err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    tp_ctx *c = tp_load(gguf.c_str(), device.c_str(), 0, "f32");
    if (!c) return 1;

    tap_sink no_tap;
    menc_result m;
    if (!c->menc.run(c->be, fx.image.data(), m, no_tap, c->err)) return 1;
    if (!c->mdec.built && !c->mdec.build(c->be, false, c->err)) return 1;
    if (!c->mdec.set_feats(m, c->err)) return 1;

    std::vector<float> style(512), img((size_t) 3 * 512 * 512);
    for (int i = 0; i < 512; i++) style[(size_t) i] = m.appearance[(size_t) i];

    // warmup: first 2 frames
    for (int f = 0; f < 2; f++)
        if (!c->mdec.decode(c->be, style.data(), img.data(), no_tap, c->err)) return 1;

    // also exercise one warm FMT step for the per-ODE-step claim
    std::vector<float> x60((size_t) 60 * 512, 0.0f), cond((size_t) 3 * 60 * 1031, 0.0f);
    std::vector<float> v((size_t) 3 * 60 * 512);
    c->fmt.forward(c->be, x60.data(), cond.data(), 0.5f, v.data(), no_tap, c->err);

    long a0, f0, per_frame_max = 0, total = 0;
    g_allocs = g_frees = g_bytes = 0;
    for (int f = 0; f < frames; f++) {
        g_count = true;
        a0 = g_allocs.load();
        if (!c->mdec.decode(c->be, style.data(), img.data(), no_tap, c->err)) return 1;
        g_count = false;
        const long d = g_allocs.load() - a0;
        total += d;
        if (d > per_frame_max) per_frame_max = d;
    }
    long fmt_allocs;
    {
        g_count = true;
        a0 = g_allocs.load();
        c->fmt.forward(c->be, x60.data(), cond.data(), 0.5f, v.data(), no_tap, c->err);
        g_count = false;
        fmt_allocs = g_allocs.load() - a0;
    }

    FILE *fo = fopen(out.c_str(), "w");
    if (!fo) return 1;
    const bool pass = total == 0 && fmt_allocs == 0;
    fprintf(fo,
            "%s\ncounting-allocator hook (malloc/calloc/realloc + operator new via malloc)\n"
            "device=%s frames=%d (after 2-frame warmup)\n"
            "heap allocations per frame: total=%ld max_per_frame=%ld\n"
            "heap allocations per FMT/ODE step (warm): %ld\n"
            "frees during loop: %ld\n",
            pass ? "PASS" : "FAIL", device.c_str(), frames, total, per_frame_max,
            fmt_allocs, g_frees.load());
    fclose(fo);
    printf("%s: total=%ld max/frame=%ld fmt_step=%ld -> %s\n",
           pass ? "PASS" : "FAIL", total, per_frame_max, fmt_allocs, out.c_str());
    tp_free(c);
    return pass ? 0 : 1;
}
