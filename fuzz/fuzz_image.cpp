// image loader fuzz (stb_image + crop/resize + normalize)
#include "tp.h"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    char err[128];
    float *img = tp_load_image(data, size, err, sizeof err);
    if (img) {
        volatile float acc = 0;
        for (size_t i = 0; i < (size_t) 3 * 512 * 512; i += 4093) acc += img[i];
        (void) acc;
        tp_buf_free(img);
    }
    return 0;
}
