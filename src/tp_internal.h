// internal shared infrastructure: backend, weight loading, compute plans,
// tap capture. Not installed; everything here is C++ and may not cross tp.h.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tp {

// ---------------- backend ----------------
struct backend_t {
    ggml_backend_t be = nullptr;
    bool gpu = false;
    int n_threads = 0;

    // device: nullptr|"cpu"|"gpu"|"cuda" (registry pick, trellis2 pattern)
    bool init(const char *device, int n_threads, std::string &err);
    void release();
    ~backend_t() { release(); }
};

// ---------------- weights ----------------
// GGUF loader (llama.cpp streaming pattern). All tensors are materialized as
// F32 on the backend (f16 files are dequantized at load); per-tensor scale
// folding (Equal* runtime scales) happens on the host before upload.
struct weights_t {
    ggml_context *ctx = nullptr;               // tensor metadata
    ggml_backend_buffer_t buf = nullptr;       // backend storage
    std::map<std::string, ggml_tensor *> map;  // name -> tensor
    std::map<std::string, std::string> kv_str;
    std::map<std::string, double> kv_num;
    bool keep_f16 = false;  // keep file-f16 tensors as F16 (fast path);
                            // default dequantizes everything to F32
    std::string f16_exclude_prefix;  // names with this prefix stay F32

    // scale_for(name, ne): multiplier folded into the tensor at load
    // (Equal* runtime scales); ne is ggml order (ne0 = torch last dim)
    bool load(const std::string &path, backend_t &be, const char *expect_arch,
              const std::function<double(const std::string &, const int64_t *)> &scale_for,
              std::string &err);
    // host-side f32 copy hook: tensors whose data is also needed on the host
    // (precomputed reductions). filled during load for names in want_host.
    std::vector<std::string> want_host;
    std::map<std::string, std::vector<float>> host;

    ggml_tensor *get(const std::string &name) const {
        auto it = map.find(name);
        return it == map.end() ? nullptr : it->second;
    }
    double num(const std::string &key, double dflt) const {
        auto it = kv_num.find(key);
        return it == kv_num.end() ? dflt : it->second;
    }
    void release();
    ~weights_t() { release(); }
};

// ---------------- compute plan ----------------
// A graph built once and computed many times (zero steady-state allocation):
// begin() -> build ops on .ctx -> mark inputs/outputs -> commit() -> loop
// { set inputs; compute(); read outputs }.
struct plan_t {
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    size_t graph_size = 0;

    void begin(size_t max_nodes);
    ggml_tensor *input(const char *name, ggml_tensor *t) {
        ggml_set_name(t, name);
        ggml_set_input(t);
        named[name] = t;
        return t;
    }
    ggml_tensor *output(const char *name, ggml_tensor *t) {
        ggml_set_name(t, name);
        ggml_set_output(t);
        named[name] = t;
        outputs.push_back(t);
        return t;
    }
    bool commit(backend_t &be, std::string &err);  // expand outputs + alloc
    bool compute(backend_t &be, std::string &err);
    ggml_tensor *operator[](const char *name) const {
        auto it = named.find(name);
        return it == named.end() ? nullptr : it->second;
    }
    void set(const char *name, const void *data) {
        ggml_tensor *t = named.at(name);
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }
    void get(const char *name, void *data) const {
        ggml_tensor *t = named.at(name);
        ggml_backend_tensor_get(t, data, 0, ggml_nbytes(t));
    }
    void release();
    ~plan_t() { release(); }

    std::map<std::string, ggml_tensor *> named;
    std::vector<ggml_tensor *> outputs;
};

// ---------------- tap sink ----------------
// Milestone parity taps: subnets register named graph tensors during build;
// after compute (before the plan is released) they call harvest(), which
// reads each tensor back and writes it through the dump writer.
struct tap_writer;  // dump.h
struct tap_entry {
    std::string name;
    ggml_tensor *t;
    std::vector<int64_t> shape;  // torch order
};
struct tap_sink {
    tap_writer *wr = nullptr;  // null = taps disabled
    std::string prefix;
    std::vector<tap_entry> pending;
    std::string err;

    void operator()(const std::string &name, ggml_tensor *t, std::vector<int64_t> shape) {
        if (wr) pending.push_back({prefix + name, t, std::move(shape)});
    }
    bool harvest();  // implemented in dump.cpp
    explicit operator bool() const { return wr != nullptr; }
};

// ---------------- misc ----------------
bool read_file(const std::string &path, std::vector<uint8_t> &out, std::string &err);
bool read_f32_file(const std::string &path, std::vector<float> &out, std::string &err);
bool write_file(const std::string &path, const void *data, size_t size, std::string &err);

inline int64_t nelems(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// layer-norm along ne0 with affine params (w,b may be null)
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w,
                        ggml_tensor *b, float eps);
// y = W x + b for torch Linear weights (W ggml [in, out])
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *W, ggml_tensor *b, ggml_tensor *x);

}  // namespace tp
