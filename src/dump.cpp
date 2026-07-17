#include "dump.h"

#include "tp_internal.h"
#include "tp.h"

#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

namespace tp {

static const char *EMOTIONS[TP_N_EMOTIONS] = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"};

static bool mkdirs(const std::string &path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && cur != "." && cur != "..") {
                mkdir(cur.c_str(), 0755);  // EEXIST is fine
            }
        }
        if (i < path.size()) cur += path[i];
    }
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool add_entry(jval &taps, const std::string &name, const char *dtype,
                      const std::vector<int64_t> &shape) {
    jval e;
    e.kind = jval::OBJ;
    jval sh;
    sh.kind = jval::ARR;
    for (int64_t d : shape) {
        jval n;
        n.kind = jval::NUM;
        n.num = (double) d;
        sh.arr.push_back(n);
    }
    jval dt;
    dt.kind = jval::STR;
    dt.str = dtype;
    e.obj.emplace_back("shape", std::move(sh));
    e.obj.emplace_back("dtype", std::move(dt));
    taps.obj.emplace_back(name, std::move(e));
    return true;
}

bool tap_writer::add_f32(const std::string &name, const float *data,
                         const std::vector<int64_t> &torch_shape) {
    if (!mkdirs(dir)) {
        err = "cannot create " + dir;
        return false;
    }
    if (!write_file(dir + "/" + name + ".f32", data, (size_t) nelems(torch_shape) * 4, err))
        return false;
    return add_entry(taps, name, "f32", torch_shape);
}

bool tap_writer::add_i32(const std::string &name, const int32_t *data,
                         const std::vector<int64_t> &torch_shape) {
    if (!mkdirs(dir)) {
        err = "cannot create " + dir;
        return false;
    }
    if (!write_file(dir + "/" + name + ".i32", data, (size_t) nelems(torch_shape) * 4, err))
        return false;
    return add_entry(taps, name, "i32", torch_shape);
}

bool tap_writer::write_meta(const std::string &case_name) {
    jval root;
    root.kind = jval::OBJ;
    jval c;
    c.kind = jval::STR;
    c.str = case_name;
    root.obj.emplace_back("case", std::move(c));
    jval port;
    port.kind = jval::STR;
    port.str = "cpp";
    root.obj.emplace_back("port", std::move(port));
    root.obj.emplace_back("taps", taps);
    const std::string text = json_write(root, 1);
    return write_file(dir + "/meta.json", text.data(), text.size(), err);
}

bool fixture::load(const std::string &case_dir, std::string &err) {
    std::vector<uint8_t> raw;
    if (!read_file(case_dir + "/meta.json", raw, err)) return false;
    jval meta;
    if (!json_parse(std::string(raw.begin(), raw.end()), meta, err)) {
        err = case_dir + "/meta.json: " + err;
        return false;
    }
    case_name = meta.str_or("case", "");
    if (!json_i64(meta.num_or("n_frames", 0), 1, 1000000, n_frames) ||
        !json_i64(meta.num_or("n_chunks", 0), 1, 100000, n_chunks)) {
        err = case_dir + ": bad meta.json (n_frames/n_chunks out of range)";
        return false;
    }
    const std::string emo = meta.str_or("emotion", "");
    for (int i = 0; i < TP_N_EMOTIONS; i++)
        if (emo == EMOTIONS[i]) emotion = i;
    if (n_frames <= 0 || n_chunks <= 0 || emotion < 0) {
        err = case_dir + ": bad meta.json (n_frames/n_chunks/emotion)";
        return false;
    }
    if (!read_f32_file(case_dir + "/image_512.f32", image, err)) return false;
    if (!read_f32_file(case_dir + "/audio_16k.f32", audio, err)) return false;
    if (!read_f32_file(case_dir + "/ode_noise.f32", noise, err)) return false;
    n_samples = (int64_t) audio.size();
    if (image.size() != 3u * 512 * 512 ||
        noise.size() != (size_t) n_chunks * 50 * 512) {
        err = case_dir + ": fixture size mismatch";
        return false;
    }
    return true;
}

bool tap_sink::harvest() {
    if (!wr) {
        pending.clear();
        return true;
    }
    for (auto &e : pending) {
        const int64_t n = nelems(e.shape);
        if (n != ggml_nelements(e.t)) {
            err = "tap " + e.name + ": shape/count mismatch";
            return false;
        }
        std::vector<float> buf((size_t) n);
        ggml_backend_tensor_get(e.t, buf.data(), 0, (size_t) n * 4);
        if (!wr->add_f32(e.name, buf.data(), e.shape)) {
            err = wr->err;
            return false;
        }
    }
    pending.clear();
    return true;
}

bool load_ref_tap(const std::string &case_dir, const std::string &name,
                  std::vector<float> &out, std::vector<int64_t> &shape,
                  std::string &err) {
    std::vector<uint8_t> raw;
    if (!read_file(case_dir + "/meta.json", raw, err)) return false;
    jval meta;
    if (!json_parse(std::string(raw.begin(), raw.end()), meta, err)) return false;
    const jval *taps = meta.find("taps");
    const jval *e = taps ? taps->find(name) : nullptr;
    if (!e) {
        err = "tap not found in reference meta: " + name;
        return false;
    }
    shape.clear();
    const jval *sh = e->find("shape");
    if (!sh || sh->kind != jval::ARR || sh->arr.size() > 8) {
        err = "bad shape for " + name;
        return false;
    }
    int64_t count = 1;
    for (auto &d : sh->arr) {
        int64_t dim = 0;
        if (d.kind != jval::NUM || !json_i64(d.num, 0, 1 << 24, dim) ||
            count > (int64_t) 1 << 31 || (dim && count > ((int64_t) 1 << 31) / dim)) {
            err = "shape out of range for " + name;
            return false;
        }
        count *= dim;
        shape.push_back(dim);
    }
    if (!read_f32_file(case_dir + "/" + name + ".f32", out, err)) return false;
    if ((int64_t) out.size() != count) {
        err = "size mismatch for " + name;
        return false;
    }
    return true;
}

}  // namespace tp
