// portrait loader (untrusted input, fuzzed): stb_image decode ->
// center-crop -> bilinear resize to 512 -> (x/255-0.5)/0.5 normalize.
#include "tp.h"

#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_STATIC
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb/stb_image_resize2.h"

namespace {
void set_err(char *err, size_t err_size, const char *msg) {
    if (err && err_size) {
        strncpy(err, msg, err_size - 1);
        err[err_size - 1] = 0;
    }
}
}  // namespace

extern "C" float *tp_load_image(const void *data, size_t size, char *err,
                                size_t err_size) {
    if (!data || !size) {
        set_err(err, err_size, "empty image");
        return nullptr;
    }
    int w = 0, h = 0, comp = 0;
    // header-only pre-check: reject oversized images BEFORE decoding
    if (!stbi_info_from_memory((const stbi_uc *) data, (int) size, &w, &h, &comp) ||
        w < 8 || h < 8 || w > (1 << 13) || h > (1 << 13) ||
        (int64_t) w * h > (int64_t) 1 << 24) {
        set_err(err, err_size, "image missing/oversized/undersized");
        return nullptr;
    }
    unsigned char *rgb = stbi_load_from_memory((const stbi_uc *) data, (int) size,
                                               &w, &h, &comp, 3);
    if (!rgb) {
        set_err(err, err_size, "image decode failed");
        return nullptr;
    }
    if (w < 8 || h < 8 || w > 1 << 14 || h > 1 << 14) {
        stbi_image_free(rgb);
        set_err(err, err_size, "image dimensions out of range");
        return nullptr;
    }
    // center-crop to square
    const int side = w < h ? w : h;
    const int x0 = (w - side) / 2, y0 = (h - side) / 2;
    unsigned char *sq = (unsigned char *) malloc((size_t) side * side * 3);
    if (!sq) {
        stbi_image_free(rgb);
        set_err(err, err_size, "out of memory");
        return nullptr;
    }
    for (int y = 0; y < side; y++)
        memcpy(sq + (size_t) y * side * 3, rgb + ((size_t) (y + y0) * w + x0) * 3,
               (size_t) side * 3);
    stbi_image_free(rgb);

    unsigned char *rs = (unsigned char *) malloc((size_t) 512 * 512 * 3);
    if (!rs) {
        free(sq);
        set_err(err, err_size, "out of memory");
        return nullptr;
    }
    if (!stbir_resize_uint8_linear(sq, side, side, 0, rs, 512, 512, 0, STBIR_RGB)) {
        free(sq);
        free(rs);
        set_err(err, err_size, "resize failed");
        return nullptr;
    }
    free(sq);

    float *out = (float *) malloc((size_t) 3 * 512 * 512 * sizeof(float));
    if (!out) {
        free(rs);
        set_err(err, err_size, "out of memory");
        return nullptr;
    }
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 512 * 512; i++)
            out[(size_t) c * 512 * 512 + i] =
                ((float) rs[(size_t) i * 3 + c] / 255.0f - 0.5f) / 0.5f;
    free(rs);
    return out;
}
