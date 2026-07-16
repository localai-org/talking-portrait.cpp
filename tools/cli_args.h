// tp-cli argument parsing, IO-free so the fuzzer can drive it (M7: CLI arg
// paths are an untrusted-input surface).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace tp {

struct cli_opts {
    std::string gguf, image, wav, emotion = "neutral", out_mp4, out_frames;
    std::string device = "gpu", ftype = "f32";
    uint64_t seed = 15;
    int threads = 0;
};

// returns false + err on bad args; performs no IO
inline bool cli_parse(int argc, const char *const *argv, cli_opts &o, std::string &err) {
    for (int i = 1; i < argc; i++) {
        if (!argv[i]) {
            err = "null argument";
            return false;
        }
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc || !argv[i + 1]) return nullptr;
            return argv[++i];
        };
        const char *v = nullptr;
        if (a == "--gguf") {
            if (!(v = next())) break;
            o.gguf = v;
        } else if (a == "--image") {
            if (!(v = next())) break;
            o.image = v;
        } else if (a == "--wav") {
            if (!(v = next())) break;
            o.wav = v;
        } else if (a == "--emotion") {
            if (!(v = next())) break;
            o.emotion = v;
        } else if (a == "--out") {
            if (!(v = next())) break;
            o.out_mp4 = v;
        } else if (a == "--frames") {
            if (!(v = next())) break;
            o.out_frames = v;
        } else if (a == "--device") {
            if (!(v = next())) break;
            o.device = v;
        } else if (a == "--ftype") {
            if (!(v = next())) break;
            o.ftype = v;
        } else if (a == "--seed") {
            if (!(v = next())) break;
            o.seed = strtoull(v, nullptr, 10);
        } else if (a == "--threads") {
            if (!(v = next())) break;
            long t = strtol(v, nullptr, 10);
            o.threads = t < -(1 << 20) ? 0 : (t > (1 << 20) ? 0 : (int) t);
        } else {
            err = "unknown arg: " + a;
            return false;
        }
        if (a.size() >= 2 && a[0] == '-' && !v && err.empty() && a != "--help") {
            err = "missing value for " + a;
            return false;
        }
    }
    if (o.gguf.empty() || o.image.empty() || o.wav.empty() ||
        (o.out_mp4.empty() && o.out_frames.empty())) {
        err = "usage: tp-cli --gguf DIR --image IMG --wav WAV [--emotion E] "
              "(--out X.mp4 | --frames X.f32) [--device cpu|gpu] [--ftype f32|f16] "
              "[--seed N] [--threads N]";
        return false;
    }
    if (o.device != "cpu" && o.device != "gpu" && o.device != "cuda" && o.device != "auto") {
        err = "bad --device";
        return false;
    }
    if (o.ftype != "f32" && o.ftype != "f16") {
        err = "bad --ftype";
        return false;
    }
    return true;
}

}  // namespace tp
