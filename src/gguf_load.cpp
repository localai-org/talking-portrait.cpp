// GGUF v3 reader matching the harness converter conventions (ggufio.py):
// verbatim tensor names, dims reversed from torch (may include a trailing
// unit dim -> 5 entries for the 5-D modconv weights, which stock ggml's
// reader rejects; we squeeze unit dims beyond GGML_MAX_DIMS), alignment 32,
// value types: u32/f32/bool/string. All tensors materialize as F32 on the
// backend; per-tensor scale folding (Equal* runtime scales) happens here.
#include "tp_internal.h"

#include <cstdio>
#include <fstream>

namespace tp {

namespace {

struct reader {
    const uint8_t *p, *end;
    bool ok = true;
    template <typename T> T rd() {
        T v{};
        if (end - p < (ptrdiff_t) sizeof(T)) {
            ok = false;
            return v;
        }
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = rd<uint64_t>();
        if (!ok || n > (uint64_t) (end - p) || n > (1u << 20)) {
            ok = false;
            return {};
        }
        std::string s((const char *) p, (size_t) n);
        p += n;
        return s;
    }
};

struct tinfo {
    std::string name;
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    uint32_t gtype = 0;
    uint64_t offset = 0;
    int64_t n = 1;
};

}  // namespace

bool weights_t::load(const std::string &path, backend_t &be, const char *expect_arch,
                     const std::function<double(const std::string &, const int64_t *)> &scale_for,
                     std::string &err) {
    std::vector<uint8_t> data;
    if (!read_file(path, data, err)) return false;
    reader r{data.data(), data.data() + data.size()};

    char magic[4];
    for (char &c : magic) c = (char) r.rd<uint8_t>();
    if (!r.ok || memcmp(magic, "GGUF", 4) != 0) {
        err = path + ": not a GGUF file";
        return false;
    }
    const uint32_t version = r.rd<uint32_t>();
    if (version != 3) {
        err = path + ": unsupported GGUF version";
        return false;
    }
    const uint64_t n_tensors = r.rd<uint64_t>();
    const uint64_t n_kv = r.rd<uint64_t>();
    if (!r.ok || n_tensors > 100000 || n_kv > 100000) {
        err = path + ": corrupt header";
        return false;
    }

    for (uint64_t i = 0; i < n_kv && r.ok; i++) {
        std::string key = r.str();
        const uint32_t vt = r.rd<uint32_t>();
        switch (vt) {
            case 4: kv_num[key] = r.rd<uint32_t>(); break;         // u32
            case 5: kv_num[key] = r.rd<int32_t>(); break;          // i32
            case 6: kv_num[key] = r.rd<float>(); break;            // f32
            case 7: kv_num[key] = r.rd<uint8_t>() ? 1 : 0; break;  // bool
            case 8: kv_str[key] = r.str(); break;                  // string
            default:
                err = path + ": unsupported KV type " + std::to_string(vt) + " for " + key;
                return false;
        }
    }
    if (!r.ok) {
        err = path + ": truncated KV section";
        return false;
    }
    if (expect_arch) {
        auto it = kv_str.find("general.architecture");
        if (it == kv_str.end() || it->second != expect_arch) {
            err = path + ": unexpected architecture '" +
                  (it == kv_str.end() ? "<none>" : it->second) + "', want " + expect_arch;
            return false;
        }
    }

    std::vector<tinfo> infos((size_t) n_tensors);
    for (auto &t : infos) {
        t.name = r.str();
        const uint32_t nd = r.rd<uint32_t>();
        if (!r.ok || nd > 8) {
            err = path + ": corrupt tensor info";
            return false;
        }
        int used = 0;
        for (uint32_t j = 0; j < nd; j++) {
            const uint64_t d = r.rd<uint64_t>();
            if (d != 1) {
                if (used >= GGML_MAX_DIMS) {
                    err = path + ": tensor " + t.name + " has >4 non-unit dims";
                    return false;
                }
                t.ne[used++] = (int64_t) d;
            } else if (used < GGML_MAX_DIMS) {
                t.ne[used++] = 1;  // keep unit dims in order while room
            }
            t.n *= (int64_t) (d ? d : 1);
        }
        t.gtype = r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        if (t.gtype > 1) {
            err = path + ": tensor " + t.name + " has unsupported type";
            return false;
        }
    }
    if (!r.ok) {
        err = path + ": truncated tensor info section";
        return false;
    }
    const size_t pre = (size_t) (r.p - data.data());
    const size_t data_off = (pre + 31) & ~(size_t) 31;

    ggml_init_params cp = {
        /*mem_size  =*/ ggml_tensor_overhead() * (infos.size() + 1),
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    ctx = ggml_init(cp);
    for (auto &t : infos) {
        ggml_tensor *d = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, t.ne);
        ggml_set_name(d, t.name.c_str());
        map[t.name] = d;
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, be.be);
    if (!buf) {
        err = "failed to allocate weight buffer for " + path;
        return false;
    }

    std::vector<float> f32;
    for (auto &t : infos) {
        const size_t esz = t.gtype == 0 ? 4 : 2;
        const size_t off = data_off + t.offset;
        if (off + (size_t) t.n * esz > data.size()) {
            err = path + ": tensor " + t.name + " out of bounds";
            return false;
        }
        f32.resize((size_t) t.n);
        if (t.gtype == 0) {
            memcpy(f32.data(), data.data() + off, (size_t) t.n * 4);
        } else {
            const ggml_fp16_t *h = (const ggml_fp16_t *) (data.data() + off);
            for (int64_t j = 0; j < t.n; j++) f32[(size_t) j] = ggml_fp16_to_fp32(h[j]);
        }
        const double s = scale_for ? scale_for(t.name, t.ne) : 1.0;
        if (s != 1.0) {
            const float fs = (float) s;
            for (int64_t j = 0; j < t.n; j++) f32[(size_t) j] *= fs;
        }
        ggml_backend_tensor_set(map[t.name], f32.data(), 0, (size_t) t.n * 4);
        for (const std::string &wname : want_host) {
            if (wname == t.name) host[t.name] = f32;
        }
    }
    return true;
}

void weights_t::release() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
    buf = nullptr;
    ctx = nullptr;
    map.clear();
}

}  // namespace tp
