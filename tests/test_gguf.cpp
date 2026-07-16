#undef NDEBUG  // asserts are the test
// weight-loading smoke: parses all three GGUFs (env TP_GGUF_DIR) on CPU
#include "audio_enc.h"
#include "fmt.h"
#include "motion_dec.h"
#include "motion_enc.h"
#include "tp_internal.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace tp;

int main() {
    const char *dir = getenv("TP_GGUF_DIR");
    if (!dir) {
        printf("SKIP: TP_GGUF_DIR not set\n");
        return 77;
    }
    std::string err;
    backend_t be;
    assert(be.init("cpu", 2, err));

    audio_model am;
    if (!am.load(std::string(dir) + "/audio_f32.gguf", be, err)) {
        printf("audio: %s\n", err.c_str());
        return 1;
    }
    assert(am.n_layers == 12 && am.hidden == 768);
    assert(am.w.map.size() == 213);

    fmt_model fm;
    if (!fm.load(std::string(dir) + "/fmt_f32.gguf", be, err)) {
        printf("fmt: %s\n", err.c_str());
        return 1;
    }
    assert(fm.depth == 8 && fm.dim_h == 1024);
    assert(fm.w.get("pos_embed"));
    // mask row 0: attends cols 0..2
    assert(fm.mask_i32[0] == 0 && fm.mask_i32[2] == 0 && fm.mask_i32[3] == 1);
    // mask row 5: cols 3..7
    assert(fm.mask_i32[5 * 60 + 2] == 1 && fm.mask_i32[5 * 60 + 3] == 0 &&
           fm.mask_i32[5 * 60 + 7] == 0 && fm.mask_i32[5 * 60 + 8] == 1);

    weights_t mw;
    mw.want_host = motion_dec_host_tensors();
    if (!mw.load(std::string(dir) + "/motion_ae_f32.gguf", be, "float-motion-ae",
                 motion_ae_scale, err)) {
        printf("motion_ae: %s\n", err.c_str());
        return 1;
    }
    // 5-D modconv weight must squeeze to [3,3,IC,OC]
    ggml_tensor *t = mw.get("dec.convs.0.conv.weight");
    assert(t && t->ne[0] == 3 && t->ne[1] == 3 && t->ne[2] == 512 && t->ne[3] == 512);
    ggml_tensor *q = mw.get("dec.direction.Q");
    assert(q && q->ne[0] == 20 && q->ne[1] == 512);
    assert(mw.host.count("dec.conv1.conv.weight"));

    printf("test_gguf OK (%zu audio, %zu fmt, %zu motion tensors)\n",
           am.w.map.size(), fm.w.map.size(), mw.map.size());
    return 0;
}
