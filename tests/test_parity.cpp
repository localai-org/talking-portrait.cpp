#undef NDEBUG  // asserts are the test
// quick internal parity smoke (the harness gate is authoritative):
// audio path aud.proj + menc.motion vs reference taps on a-short.
// env: TP_GGUF_DIR, TP_REF_DUMPS
#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace tp;

static void row_metrics(const std::vector<float> &ref, const std::vector<float> &got,
                        size_t row, double &cos_min, double &err_max) {
    const size_t n = ref.size() / row;
    cos_min = 1.0;
    err_max = 0.0;
    for (size_t r = 0; r < n; r++) {
        double dot = 0, na = 0, nb = 0, me = 0, mr = 0;
        for (size_t i = 0; i < row; i++) {
            const double a = ref[r * row + i], b = got[r * row + i];
            dot += a * b;
            na += a * a;
            nb += b * b;
            me = std::max(me, fabs(a - b));
            mr = std::max(mr, fabs(a));
        }
        const double c = (na < 1e-30 && nb < 1e-30) ? 1.0 : dot / std::max(sqrt(na) * sqrt(nb), 1e-30);
        cos_min = std::min(cos_min, c);
        err_max = std::max(err_max, me / std::max(1.0, mr));
    }
}

int main() {
    const char *gguf = getenv("TP_GGUF_DIR");
    const char *refs = getenv("TP_REF_DUMPS");
    if (!gguf || !refs) {
        printf("SKIP: TP_GGUF_DIR/TP_REF_DUMPS not set\n");
        return 77;
    }
    const std::string case_dir = std::string(refs) + "/a-short";
    std::string err;
    fixture fx;
    if (!fx.load(case_dir, err)) {
        printf("SKIP: %s\n", err.c_str());
        return 77;
    }
    tp_ctx *c = tp_load(gguf, "cpu", 0, "f32");
    if (!c) return 1;

    tap_sink no_tap;
    std::vector<float> wa;
    assert(c->audio.encode(c->be, fx.audio.data(), fx.n_samples, fx.n_frames, wa,
                           no_tap, c->err));
    std::vector<float> ref;
    std::vector<int64_t> sh;
    assert(load_ref_tap(case_dir, "aud.proj", ref, sh, err));
    double cs, ce;
    row_metrics(ref, wa, 512, cs, ce);
    printf("aud.proj: cos_min=%.7f err_max=%.2e\n", cs, ce);
    assert(cs > 0.9999 && ce < 2e-2);

    menc_result m;
    assert(c->menc.run(c->be, fx.image.data(), m, no_tap, c->err));
    assert(load_ref_tap(case_dir, "menc.motion", ref, sh, err));
    row_metrics(ref, m.motion, 512, cs, ce);
    printf("menc.motion: cos_min=%.7f err_max=%.2e\n", cs, ce);
    assert(cs > 0.9999 && ce < 2e-2);

    tp_free(c);
    printf("test_parity OK\n");
    return 0;
}
