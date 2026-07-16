/* talking-portrait.cpp — C API
 *
 * C++/GGML port of FLOAT (audio-driven talking portrait):
 * one 512x512 portrait + 16 kHz speech -> 25 fps talking-head frames.
 *
 * Conventions (privacy-filter.cpp lineage): no exceptions cross this
 * boundary, every pointer-returning function reports failure via
 * tp_last_error, every free is NULL-safe. Opaque handle, caller-owned flat
 * buffers, fixed-width types — shaped for FFI.
 */
#ifndef TP_H
#define TP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TP_ABI_VERSION 1

typedef struct tp_ctx tp_ctx;

/* emotion labels, index order: angry,disgust,fear,happy,neutral,sad,surprise */
#define TP_N_EMOTIONS 7

int tp_abi_version(void);

/* Load the three sub-net GGUFs from a directory containing
 * audio_{f32,f16}.gguf, fmt_*.gguf, motion_ae_*.gguf.
 * device: NULL|"cpu"|"gpu"|"cuda" ; n_threads<=0 = auto ; ftype: "f32"|"f16".
 * Returns NULL on failure (no context to query — check stderr). */
tp_ctx *tp_load(const char *gguf_dir, const char *device, int n_threads,
                const char *ftype);
void tp_free(tp_ctx *ctx);
const char *tp_last_error(const tp_ctx *ctx);

/* ---- end-to-end generation ----
 * image : [3*512*512] f32, CHW, normalized to [-1,1]
 * audio : [n_samples] f32 mono 16 kHz, already zero-mean/unit-var normalized
 *         (tp_normalize_audio) — parity fixtures come pre-normalized
 * emotion: index into TP_N_EMOTIONS (one-hot condition)
 * noise : optional [n_chunks*50*512] f32 x0 per chunk (parity fixtures);
 *         NULL = draw from seed
 * out   : caller buffer [n_frames*3*512*512] f32 in [-1,1]
 * n_frames = ceil(n_samples * 25 / 16000); returns n_frames or -1 on error */
int64_t tp_n_frames(int64_t n_samples);
int64_t tp_n_chunks(int64_t n_frames);
int tp_generate(tp_ctx *ctx, const float *image, const float *audio,
                int64_t n_samples, int emotion, const float *noise,
                uint64_t seed, float *out_frames);

/* per-frame streaming variant: cb is called with each decoded frame
 * (frame data valid only during the call); return nonzero from cb to stop. */
typedef int (*tp_frame_cb)(int64_t frame_idx, const float *chw, void *ud);
int tp_generate_cb(tp_ctx *ctx, const float *image, const float *audio,
                   int64_t n_samples, int emotion, const float *noise,
                   uint64_t seed, tp_frame_cb cb, void *ud);

/* ---- input helpers (untrusted-input surfaces, fuzzed) ---- */
/* wav: PCM16/f32 RIFF, mono/stereo; must be 16 kHz. returns malloc'd
 * samples (caller frees with tp_buf_free) or NULL. */
float *tp_load_wav(const void *data, size_t size, int64_t *n_samples,
                   char *err, size_t err_size);
/* image file (png/jpg/...) -> [3*512*512] f32 normalized to [-1,1]
 * (aspect-preserving center-crop + bilinear resize). */
float *tp_load_image(const void *data, size_t size, char *err, size_t err_size);
/* zero-mean/unit-variance over the clip (Wav2Vec2FeatureExtractor). */
void tp_normalize_audio(float *audio, int64_t n_samples);
int tp_emotion_index(const char *label); /* -1 if unknown */
void tp_buf_free(void *p);

#ifdef __cplusplus
}
#endif
#endif /* TP_H */
