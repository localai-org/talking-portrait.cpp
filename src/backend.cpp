#include "tp_internal.h"

#include <cstdio>
#include <fstream>
#include <thread>

namespace tp {

bool backend_t::init(const char *device, int nth, std::string &err) {
    const std::string dev = device ? device : "auto";
    if (dev != "cpu") {
        // device registry walk (trellis2 pattern): first GPU device wins
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                be = ggml_backend_dev_init(d, nullptr);
                if (be) {
                    gpu = true;
                    return true;
                }
            }
        }
        if (dev == "gpu" || dev == "cuda") {
            err = "no GPU backend available";
            return false;
        }
    }
    be = ggml_backend_cpu_init();
    if (!be) {
        err = "failed to init CPU backend";
        return false;
    }
    n_threads = nth > 0 ? nth : (int) std::thread::hardware_concurrency();
    if (n_threads < 1) n_threads = 4;
    ggml_backend_cpu_set_n_threads(be, n_threads);
    return true;
}

void backend_t::release() {
    if (be) ggml_backend_free(be);
    be = nullptr;
}

// ---------------- plan ----------------

void plan_t::begin(size_t max_nodes) {
    release();
    graph_size = max_nodes;
    ggml_init_params p = {
        /*mem_size  =*/ ggml_tensor_overhead() * max_nodes +
                        ggml_graph_overhead_custom(max_nodes, false),
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    ctx = ggml_init(p);
    gf = ggml_new_graph_custom(ctx, max_nodes, false);
}

bool plan_t::commit(backend_t &be, std::string &err) {
    for (ggml_tensor *t : outputs) ggml_build_forward_expand(gf, t);
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.be));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        err = "graph allocation failed";
        return false;
    }
    return true;
}

bool plan_t::compute(backend_t &be, std::string &err) {
    ggml_status st = ggml_backend_graph_compute(be.be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        err = "graph compute failed (status " + std::to_string((int) st) + ")";
        return false;
    }
    return true;
}

void plan_t::release() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
    galloc = nullptr;
    ctx = nullptr;
    gf = nullptr;
    named.clear();
    outputs.clear();
}

// ---------------- misc ----------------

bool read_file(const std::string &path, std::vector<uint8_t> &out, std::string &err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t) n);
    if (n && !f.read((char *) out.data(), n)) {
        err = "short read on " + path;
        return false;
    }
    return true;
}

bool read_f32_file(const std::string &path, std::vector<float> &out, std::string &err) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw, err)) return false;
    if (raw.size() % 4) {
        err = path + ": size not a multiple of 4";
        return false;
    }
    out.resize(raw.size() / 4);
    memcpy(out.data(), raw.data(), raw.size());
    return true;
}

bool write_file(const std::string &path, const void *data, size_t size, std::string &err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot open " + path + " for writing";
        return false;
    }
    if (size && !f.write((const char *) data, (std::streamsize) size)) {
        err = "short write on " + path;
        return false;
    }
    return true;
}

ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w,
                        ggml_tensor *b, float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor *linear(ggml_context *ctx, ggml_tensor *W, ggml_tensor *b, ggml_tensor *x) {
    x = ggml_mul_mat(ctx, W, x);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

}  // namespace tp
