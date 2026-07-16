// wav reader fuzz: must never crash/leak on hostile RIFF input
#include "tp.h"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char err[128];
    int64_t n = 0;
    float *s = tp_load_wav(data, size, &n, err, sizeof err);
    if (s) {
        // touch the buffer to catch bad sizing
        volatile float acc = 0;
        for (int64_t i = 0; i < n; i += 997) acc += s[i];
        (void) acc;
        tp_buf_free(s);
    }
    return 0;
}
