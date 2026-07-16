#undef NDEBUG  // asserts are the test
#include "tp.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// build a minimal PCM16 mono 16k wav in memory
static std::vector<unsigned char> make_wav(int rate, int n, int channels = 1) {
    std::vector<unsigned char> b;
    auto u32 = [&](unsigned v) {
        for (int i = 0; i < 4; i++) b.push_back((unsigned char) (v >> (8 * i)));
    };
    auto u16 = [&](unsigned v) {
        for (int i = 0; i < 2; i++) b.push_back((unsigned char) (v >> (8 * i)));
    };
    const unsigned data_size = (unsigned) (n * 2 * channels);
    b.insert(b.end(), {'R', 'I', 'F', 'F'});
    u32(36 + data_size);
    b.insert(b.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    u32(16);
    u16(1);
    u16((unsigned) channels);
    u32((unsigned) rate);
    u32((unsigned) (rate * 2 * channels));
    u16((unsigned) (2 * channels));
    u16(16);
    b.insert(b.end(), {'d', 'a', 't', 'a'});
    u32(data_size);
    for (int i = 0; i < n * channels; i++) {
        short s = (short) (10000 * sin(0.1 * i));
        u16((unsigned short) s);
    }
    return b;
}

int main() {
    char err[128];
    int64_t n = 0;
    auto wav = make_wav(16000, 1600);
    float *s = tp_load_wav(wav.data(), wav.size(), &n, err, sizeof err);
    assert(s && n == 1600);
    tp_buf_free(s);

    // wrong rate rejected
    auto bad = make_wav(44100, 100);
    s = tp_load_wav(bad.data(), bad.size(), &n, err, sizeof err);
    assert(!s);

    // stereo downmix
    auto st = make_wav(16000, 100, 2);
    s = tp_load_wav(st.data(), st.size(), &n, err, sizeof err);
    assert(s && n == 100);
    tp_buf_free(s);

    // truncated garbage must not crash
    for (size_t cut = 0; cut < wav.size(); cut += 7) {
        s = tp_load_wav(wav.data(), cut, &n, err, sizeof err);
        if (s) tp_buf_free(s);
    }

    // normalization
    std::vector<float> x(1000);
    for (int i = 0; i < 1000; i++) x[(size_t) i] = 3.0f + (float) (i % 7);
    tp_normalize_audio(x.data(), 1000);
    double mean = 0, var = 0;
    for (float f : x) mean += f;
    mean /= 1000;
    for (float f : x) var += (f - mean) * (f - mean);
    var /= 1000;
    assert(fabs(mean) < 1e-4 && fabs(var - 1.0) < 1e-3);

    printf("test_wav OK\n");
    return 0;
}
