// RIFF/WAVE reader (untrusted input, fuzzed): PCM16 / PCM24 / PCM32 / f32,
// mono or stereo (downmixed), must be 16 kHz.
#include "tp.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct rd {
    const uint8_t *p, *end;
    bool take(void *dst, size_t n) {
        if ((size_t) (end - p) < n) return false;
        memcpy(dst, p, n);
        p += n;
        return true;
    }
    bool skip(size_t n) {
        if ((size_t) (end - p) < n) return false;
        p += n;
        return true;
    }
};

void set_err(char *err, size_t err_size, const char *msg) {
    if (err && err_size) {
        strncpy(err, msg, err_size - 1);
        err[err_size - 1] = 0;
    }
}

}  // namespace

extern "C" float *tp_load_wav(const void *data, size_t size, int64_t *n_samples,
                              char *err, size_t err_size) {
    if (n_samples) *n_samples = 0;
    if (!data || size < 12) {
        set_err(err, err_size, "not a wav file");
        return nullptr;
    }
    rd r{(const uint8_t *) data, (const uint8_t *) data + size};
    char tag[4];
    uint32_t riff_size = 0;
    if (!r.take(tag, 4) || memcmp(tag, "RIFF", 4) != 0 || !r.take(&riff_size, 4) ||
        !r.take(tag, 4) || memcmp(tag, "WAVE", 4) != 0) {
        set_err(err, err_size, "not a RIFF/WAVE file");
        return nullptr;
    }

    uint16_t fmt_code = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;
    const uint8_t *pcm = nullptr;
    size_t pcm_size = 0;

    while (r.end - r.p >= 8) {
        char cid[4];
        uint32_t csz = 0;
        if (!r.take(cid, 4) || !r.take(&csz, 4)) break;
        if ((size_t) (r.end - r.p) < csz) csz = (uint32_t) (r.end - r.p);
        if (memcmp(cid, "fmt ", 4) == 0 && csz >= 16) {
            const uint8_t *f = r.p;
            memcpy(&fmt_code, f + 0, 2);
            memcpy(&channels, f + 2, 2);
            memcpy(&rate, f + 4, 4);
            memcpy(&bits, f + 14, 2);
            if (fmt_code == 0xFFFE && csz >= 26) {  // WAVE_FORMAT_EXTENSIBLE
                uint16_t sub = 0;
                memcpy(&sub, f + 24, 2);
                fmt_code = sub;
            }
            have_fmt = true;
        } else if (memcmp(cid, "data", 4) == 0) {
            pcm = r.p;
            pcm_size = csz;
        }
        if (!r.skip(csz + (csz & 1))) break;
    }

    if (!have_fmt || !pcm) {
        set_err(err, err_size, "missing fmt/data chunk");
        return nullptr;
    }
    if (channels < 1 || channels > 8) {
        set_err(err, err_size, "unsupported channel count");
        return nullptr;
    }
    if (rate != 16000) {
        set_err(err, err_size, "sample rate must be 16000 Hz");
        return nullptr;
    }
    const bool is_float = fmt_code == 3;
    if (!(fmt_code == 1 || is_float) ||
        (is_float && bits != 32) ||
        (!is_float && bits != 16 && bits != 24 && bits != 32)) {
        set_err(err, err_size, "unsupported sample format");
        return nullptr;
    }
    const size_t bytes_per = (size_t) bits / 8;
    const size_t stride = bytes_per * channels;
    if (stride == 0) {
        set_err(err, err_size, "bad frame size");
        return nullptr;
    }
    const size_t frames = pcm_size / stride;
    if (frames == 0 || frames > (size_t) 1 << 28) {
        set_err(err, err_size, "empty or oversized wav");
        return nullptr;
    }

    float *out = (float *) malloc(frames * sizeof(float));
    if (!out) {
        set_err(err, err_size, "out of memory");
        return nullptr;
    }
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (unsigned ch = 0; ch < channels; ch++) {
            const uint8_t *s = pcm + i * stride + ch * bytes_per;
            double v = 0;
            if (is_float) {
                float f;
                memcpy(&f, s, 4);
                v = f;
            } else if (bits == 16) {
                int16_t x;
                memcpy(&x, s, 2);
                v = (double) x / 32768.0;
            } else if (bits == 24) {
                int32_t x = (int32_t) ((uint32_t) s[0] | ((uint32_t) s[1] << 8) |
                                       ((uint32_t) s[2] << 16));
                if (x & 0x800000) x |= (int32_t) 0xFF000000;
                v = (double) x / 8388608.0;
            } else {
                int32_t x;
                memcpy(&x, s, 4);
                v = (double) x / 2147483648.0;
            }
            acc += v;
        }
        out[i] = (float) (acc / channels);
    }
    if (n_samples) *n_samples = (int64_t) frames;
    return out;
}
