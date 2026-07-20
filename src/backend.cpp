#include "tp_internal.h"

#include <cstdio>
#include <fstream>
#include <thread>

namespace tp {

bool backend_t::init(const char *device, int nth, std::string &err) {
    const std::string dev = device ? device : "auto";
    if (dev != "cpu") {
        // device registry walk (trellis2 pattern): prefer a discrete NVIDIA
        // device (a Vulkan build can expose several "GPU" devices on this
        // host: 5070 Ti, AMD iGPU, llvmpipe — llvmpipe self-reports as a
        // GPU-type device and must not win), else first GPU device
        ggml_backend_dev_t pick = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char *desc = ggml_backend_dev_description(d);
            if (desc && strstr(desc, "NVIDIA")) {
                pick = d;
                break;
            }
            if (!pick) pick = d;
        }
        if (pick) be = ggml_backend_dev_init(pick, nullptr);
        if (be) {
            gpu = true;
            fprintf(stderr, "tp: backend %s on %s\n", ggml_backend_name(be),
                    ggml_backend_dev_description(pick));
            // CPU fallback for ops the GPU backend cannot run (scheduler path)
            cpu_fb = ggml_backend_cpu_init();
            if (cpu_fb) {
                int nfb = nth > 0 ? nth : (int) std::thread::hardware_concurrency();
                ggml_backend_cpu_set_n_threads(cpu_fb, nfb < 1 ? 4 : nfb);
            }
            return true;
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
    if (cpu_fb) ggml_backend_free(cpu_fb);
    if (be) ggml_backend_free(be);
    cpu_fb = nullptr;
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

// the ggml_im2col_cf fast path is encoded as GGML_OP_IM2COL with a
// channels-first flag in op_params[6] (bit 1). Backends without a cf kernel
// (everything except CPU, the patched CUDA and the patched Vulkan) still
// claim IM2COL support and would silently compute the channels-last layout —
// treat the node as unsupported there so the scheduler routes it to the CPU.
static bool node_runs_natively(ggml_backend_t be, const ggml_tensor *n) {
    const bool cf_im2col =
        n->op == GGML_OP_IM2COL && (((const int32_t *) n->op_params)[6] & 2);
    if (cf_im2col || n->op == GGML_OP_BILINEAR_GATHER ||
        n->op == GGML_OP_BIAS_LRELU || n->op == GGML_OP_SUBPIX_LRELU ||
        n->op == GGML_OP_SUBPIX_LRELU_CL || n->op == GGML_OP_UPFIRDN2X ||
        n->op == GGML_OP_CLSP_MUL || n->op == GGML_OP_CLSP_LERP ||
        n->op == GGML_OP_FLOW_GRID) {
        // measurement knob: force the custom ops onto the CPU fallback even
        // on backends that have kernels for them (baseline reproduction)
        if (getenv("TP_FORCE_FALLBACK")) return false;
    }
    if (cf_im2col) {
        const char *bn = ggml_backend_name(be);
        if (strncmp(bn, "CPU", 3) != 0 && strncmp(bn, "CUDA", 4) != 0 &&
            strncmp(bn, "Vulkan", 6) != 0) return false;
    }
    return ggml_backend_supports_op(be, n);
}

bool plan_t::commit(backend_t &be, std::string &err) {
    for (ggml_tensor *t : outputs) ggml_build_forward_expand(gf, t);

    // if the primary backend can run every node, single-backend gallocr plan
    // (CUDA / CPU path, unchanged); otherwise fall back to the ggml
    // scheduler with a CPU backend for the unsupported ops (Vulkan path)
    bool full_native = true;
    if (be.cpu_fb) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            if (!node_runs_natively(be.be, ggml_graph_node(gf, i))) {
                full_native = false;
                break;
            }
        }
    }
    if (full_native) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.be));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            err = "graph allocation failed";
            return false;
        }
        return true;
    }

    ggml_backend_t bes[2] = {be.be, be.cpu_fb};
    // sched hash set must hold nodes + leafs; no public leaf-count accessor,
    // so use a generous bound (leafs = weights/consts/inputs < node budget)
    const size_t hash_size = (size_t) ggml_graph_n_nodes(gf) + graph_size + 1024;
    sched = ggml_backend_sched_new(bes, nullptr, 2,
                                   hash_size > graph_size ? hash_size : graph_size,
                                   false, false);
    if (!sched) {
        err = "failed to create backend scheduler";
        return false;
    }
    // pin the wrong-layout / unsupported nodes to the CPU backend and log
    // the fallback set once (dedup by op) so measurements are attributable
    std::map<std::string, int> fell_back;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor *n = ggml_graph_node(gf, i);
        if (!node_runs_natively(be.be, n)) {
            ggml_backend_sched_set_tensor_backend(sched, n, be.cpu_fb);
            fell_back[ggml_op_name(n->op)]++;
            if (getenv("TP_SCHED_DEBUG"))
                fprintf(stderr,
                        "tp: sched fallback node %s '%s' dst[%lld,%lld,%lld,%lld] "
                        "src0=%s src1=%s params6=%d\n",
                        ggml_op_name(n->op), n->name, (long long) n->ne[0],
                        (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        n->src[0] ? ggml_type_name(n->src[0]->type) : "-",
                        n->src[1] ? ggml_type_name(n->src[1]->type) : "-",
                        ((const int32_t *) n->op_params)[6]);
        }
    }
    for (auto &kv : fell_back)
        fprintf(stderr, "tp: sched fallback: %s x%d -> CPU\n", kv.first.c_str(), kv.second);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        err = "scheduler graph allocation failed";
        return false;
    }
    return true;
}

bool plan_t::compute(backend_t &be, std::string &err) {
    ggml_status st = sched ? ggml_backend_sched_graph_compute(sched, gf)
                           : ggml_backend_graph_compute(be.be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        err = "graph compute failed (status " + std::to_string((int) st) + ")";
        return false;
    }
    return true;
}

void plan_t::release() {
    if (sched) ggml_backend_sched_free(sched);
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
    sched = nullptr;
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
