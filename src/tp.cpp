// C ABI wrapper — no exceptions cross this boundary
#include "tp.h"

#include "motion_dec.h"
#include "motion_enc.h"
#include "tp_ctx.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>

bool tp_ctx::load(const char *gguf_dir, const char *device, int n_threads,
                  const char *ftype) {
    const std::string dir = gguf_dir ? gguf_dir : ".";
    const std::string ft = ftype ? ftype : "f32";
    if (ft != "f32" && ft != "f16") {
        err = "ftype must be f32 or f16";
        return false;
    }
    if (!be.init(device, n_threads, err)) return false;
    if (!audio.load(dir + "/audio_" + ft + ".gguf", be, err)) return false;
    if (!fmt.load(dir + "/fmt_" + ft + ".gguf", be, err)) return false;
    if (!fmt.build(be, err)) return false;
    motion_ae.want_host = tp::motion_dec_host_tensors();
    if (!motion_ae.load(dir + "/motion_ae_" + ft + ".gguf", be, "float-motion-ae",
                        tp::motion_ae_scale, err))
        return false;
    menc.w = &motion_ae;
    mdec.w = &motion_ae;
    if (!mdec.prepare(be, err)) return false;
    motion_ae.host.clear();  // host copies only needed for prepare()
    return true;
}

extern "C" {

int tp_abi_version(void) { return TP_ABI_VERSION; }

tp_ctx *tp_load(const char *gguf_dir, const char *device, int n_threads,
                const char *ftype) {
    tp_ctx *c = new (std::nothrow) tp_ctx();
    if (!c) return nullptr;
    if (!c->load(gguf_dir, device, n_threads, ftype)) {
        fprintf(stderr, "tp_load: %s\n", c->err.c_str());
        delete c;
        return nullptr;
    }
    return c;
}

void tp_free(tp_ctx *c) { delete c; }

const char *tp_last_error(const tp_ctx *c) { return c ? c->err.c_str() : "no context"; }

int64_t tp_n_frames(int64_t n_samples) {
    if (n_samples <= 0) return 0;
    return (n_samples * 25 + 15999) / 16000;  // ceil(n * fps / sr)
}

int64_t tp_n_chunks(int64_t n_frames) { return (n_frames + 49) / 50; }

struct frames_sink {
    float *out;
};
static int frames_cb(int64_t idx, const float *chw, void *ud) {
    frames_sink *s = (frames_sink *) ud;
    memcpy(s->out + (size_t) idx * 3 * 512 * 512, chw, (size_t) 3 * 512 * 512 * 4);
    return 0;
}

int tp_generate(tp_ctx *c, const float *image, const float *audio, int64_t n_samples,
                int emotion, const float *noise, uint64_t seed, float *out_frames) {
    if (!c || !image || !audio || !out_frames) return -1;
    frames_sink s{out_frames};
    return tp::generate_impl(c, image, audio, n_samples, emotion, noise, seed,
                             frames_cb, &s);
}

int tp_generate_cb(tp_ctx *c, const float *image, const float *audio, int64_t n_samples,
                   int emotion, const float *noise, uint64_t seed, tp_frame_cb cb,
                   void *ud) {
    if (!c || !image || !audio) return -1;
    return tp::generate_impl(c, image, audio, n_samples, emotion, noise, seed, cb, ud);
}

void tp_normalize_audio(float *audio, int64_t n) {
    if (!audio || n <= 0) return;
    double mean = 0;
    for (int64_t i = 0; i < n; i++) mean += audio[i];
    mean /= (double) n;
    double var = 0;
    for (int64_t i = 0; i < n; i++) {
        const double d = audio[i] - mean;
        var += d * d;
    }
    var /= (double) n;
    const double inv = 1.0 / sqrt(var + 1e-7);
    for (int64_t i = 0; i < n; i++) audio[i] = (float) ((audio[i] - mean) * inv);
}

int tp_emotion_index(const char *label) {
    static const char *EMO[TP_N_EMOTIONS] = {"angry", "disgust", "fear", "happy",
                                             "neutral", "sad", "surprise"};
    if (!label) return -1;
    for (int i = 0; i < TP_N_EMOTIONS; i++)
        if (strcmp(label, EMO[i]) == 0) return i;
    return -1;
}

void tp_buf_free(void *p) { free(p); }

}  // extern "C"
