// parity-dump interop with the float-harness (docs/TAPS.md format):
// one raw little-endian .f32/.i32 file per tap + meta.json with shapes.
#pragma once

#include "json_min.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tp {

struct tap_writer {
    std::string dir;  // parity-dumps/<case>
    jval taps;        // OBJ: name -> {shape, dtype}
    std::string err;

    explicit tap_writer(std::string d) : dir(std::move(d)) { taps.kind = jval::OBJ; }
    bool add_f32(const std::string &name, const float *data,
                 const std::vector<int64_t> &torch_shape);
    bool add_i32(const std::string &name, const int32_t *data,
                 const std::vector<int64_t> &torch_shape);
    bool write_meta(const std::string &case_name);
};

// fixture inputs for one case, read from the harness dumps/<case> dir
struct fixture {
    std::string case_name;
    int64_t n_frames = 0, n_chunks = 0;
    int emotion = -1;                 // label index
    std::vector<float> image;         // [1,3,512,512]
    std::vector<float> audio;         // [1,N]
    std::vector<float> noise;         // [n_chunks,50,512]
    int64_t n_samples = 0;

    bool load(const std::string &case_dir, std::string &err);
};

// read one reference tap (f32) with its shape from a harness case dir
bool load_ref_tap(const std::string &case_dir, const std::string &name,
                  std::vector<float> &out, std::vector<int64_t> &shape,
                  std::string &err);

}  // namespace tp
