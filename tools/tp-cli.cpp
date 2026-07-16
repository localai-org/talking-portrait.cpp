// tp-cli: portrait + wav + emotion -> raw frames (.f32) and/or mp4 (ffmpeg)
//
//   tp-cli --gguf DIR --image face.jpg --wav speech.wav --emotion happy \
//          [--out out.mp4] [--frames out_frames.f32] [--device cpu|gpu]
//          [--ftype f32|f16] [--seed N] [--threads N]
#include "tp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_all(const char *path, std::vector<unsigned char> &out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t) n);
    const bool ok = n == 0 || fread(out.data(), 1, (size_t) n, f) == (size_t) n;
    fclose(f);
    return ok;
}

struct mp4_writer {
    FILE *pipe = nullptr;
    bool open(const std::string &path, const std::string &wav) {
        char cmd[1024];
        snprintf(cmd, sizeof cmd,
                 "ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s 512x512 -r 25 "
                 "-i - -i '%s' -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '%s'",
                 wav.c_str(), path.c_str());
        pipe = popen(cmd, "w");
        return pipe != nullptr;
    }
    void frame(const float *chw) {
        static std::vector<unsigned char> rgb((size_t) 512 * 512 * 3);
        for (int i = 0; i < 512 * 512; i++) {
            for (int c = 0; c < 3; c++) {
                float v = chw[(size_t) c * 512 * 512 + i];
                v = v < -1 ? -1 : (v > 1 ? 1 : v);
                rgb[(size_t) i * 3 + c] = (unsigned char) ((v + 1.0f) / 2.0f * 255.0f);
            }
        }
        fwrite(rgb.data(), 1, rgb.size(), pipe);
    }
    int close() { return pipe ? pclose(pipe) : 0; }
};

int main(int argc, char **argv) {
    std::string gguf, image, wav, emotion = "neutral", out_mp4, out_frames;
    std::string device = "gpu", ftype = "f32";
    uint64_t seed = 15;
    int threads = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--gguf") gguf = next();
        else if (a == "--image") image = next();
        else if (a == "--wav") wav = next();
        else if (a == "--emotion") emotion = next();
        else if (a == "--out") out_mp4 = next();
        else if (a == "--frames") out_frames = next();
        else if (a == "--device") device = next();
        else if (a == "--ftype") ftype = next();
        else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
        else if (a == "--threads") threads = atoi(next());
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || image.empty() || wav.empty() || (out_mp4.empty() && out_frames.empty())) {
        fprintf(stderr,
                "usage: tp-cli --gguf DIR --image IMG --wav WAV [--emotion E] "
                "(--out X.mp4 | --frames X.f32) [--device cpu|gpu] [--seed N]\n");
        return 2;
    }
    const int emo = tp_emotion_index(emotion.c_str());
    if (emo < 0) {
        fprintf(stderr, "unknown emotion '%s'\n", emotion.c_str());
        return 2;
    }

    std::vector<unsigned char> img_bytes, wav_bytes;
    if (!read_all(image.c_str(), img_bytes)) {
        fprintf(stderr, "cannot read %s\n", image.c_str());
        return 1;
    }
    if (!read_all(wav.c_str(), wav_bytes)) {
        fprintf(stderr, "cannot read %s\n", wav.c_str());
        return 1;
    }
    char err[256];
    float *img = tp_load_image(img_bytes.data(), img_bytes.size(), err, sizeof err);
    if (!img) {
        fprintf(stderr, "image: %s\n", err);
        return 1;
    }
    int64_t n_samples = 0;
    float *audio = tp_load_wav(wav_bytes.data(), wav_bytes.size(), &n_samples, err, sizeof err);
    if (!audio) {
        fprintf(stderr, "wav: %s\n", err);
        tp_buf_free(img);
        return 1;
    }
    tp_normalize_audio(audio, n_samples);

    tp_ctx *ctx = tp_load(gguf.c_str(), device.c_str(), threads, ftype.c_str());
    if (!ctx) {
        tp_buf_free(img);
        tp_buf_free(audio);
        return 1;
    }
    const int64_t n_frames = tp_n_frames(n_samples);
    fprintf(stderr, "generating %lld frames (%.2f s)...\n", (long long) n_frames,
            (double) n_samples / 16000.0);

    struct sink_t {
        FILE *raw = nullptr;
        mp4_writer *mp4 = nullptr;
    } sink;
    if (!out_frames.empty()) {
        sink.raw = fopen(out_frames.c_str(), "wb");
        if (!sink.raw) {
            fprintf(stderr, "cannot open %s\n", out_frames.c_str());
            return 1;
        }
    }
    mp4_writer mw;
    if (!out_mp4.empty()) {
        if (!mw.open(out_mp4, wav)) {
            fprintf(stderr, "cannot start ffmpeg\n");
            return 1;
        }
        sink.mp4 = &mw;
    }
    auto cb = [](int64_t idx, const float *chw, void *ud) -> int {
        sink_t *s = (sink_t *) ud;
        if (s->raw) fwrite(chw, 4, (size_t) 3 * 512 * 512, s->raw);
        if (s->mp4) s->mp4->frame(chw);
        if (idx % 25 == 0) fprintf(stderr, "  frame %lld\n", (long long) idx);
        return 0;
    };
    const int n = tp_generate_cb(ctx, img, audio, n_samples, emo, nullptr, seed, cb, &sink);
    if (sink.raw) fclose(sink.raw);
    if (sink.mp4) mw.close();
    if (n < 0) {
        fprintf(stderr, "generate: %s\n", tp_last_error(ctx));
        return 1;
    }
    fprintf(stderr, "done: %d frames\n", n);
    tp_free(ctx);
    tp_buf_free(img);
    tp_buf_free(audio);
    return 0;
}
