#undef NDEBUG
#include "dump.h"
#include "tp.h"
#include "tp_ctx.h"
#include <cassert>
#include <cstdio>
using namespace tp;
int main() {
    std::string err;
    fixture fx; assert(fx.load("/harness/dumps/a-short", err));
    tp_ctx *c = tp_load("/harness/ggufs", "gpu", 0, getenv("FT") ? getenv("FT") : "f32");
    assert(c);
    tap_sink nt;
    menc_result m;
    assert(c->menc.run(c->be, fx.image.data(), m, nt, c->err));
    assert(c->mdec.build(c->be, false, c->err));
    assert(c->mdec.set_feats(m, c->err));
    std::vector<float> style(512), img(3*512*512);
    for (int i = 0; i < 512; i++) style[i] = m.appearance[i];
    plan_t &p = c->mdec.frame;
    {
        std::map<std::string, int> hist;
        for (int i = 0; i < ggml_graph_n_nodes(p.gf); i++)
            hist[ggml_op_desc(ggml_graph_node(p.gf, i))]++;
        printf("graph nodes: %d\n", ggml_graph_n_nodes(p.gf));
        for (auto &kv : hist) printf("  %-24s %d\n", kv.first.c_str(), kv.second);
    }
    // warm
    for (int i = 0; i < 3; i++) { p.set("style", style.data()); p.compute(c->be, c->err); }
    ggml_backend_synchronize(c->be.be);
    const int it = getenv("PROF_ONE") ? 1 : 50;
    int64_t t0 = ggml_time_us();
    for (int i = 0; i < it; i++) p.set("style", style.data());
    printf("set style: %.3f ms\n", (ggml_time_us()-t0)/1000.0/it);
    t0 = ggml_time_us();
    for (int i = 0; i < it; i++) p.compute(c->be, c->err);
    ggml_backend_synchronize(c->be.be);
    printf("compute only: %.3f ms\n", (ggml_time_us()-t0)/1000.0/it);
    t0 = ggml_time_us();
    for (int i = 0; i < it; i++) {
        ggml_tensor *t = p["img"];
        ggml_backend_tensor_get(t, img.data(), 0, ggml_nbytes(t));
    }
    printf("get img: %.3f ms\n", (ggml_time_us()-t0)/1000.0/it);
    t0 = ggml_time_us();
    for (int i = 0; i < it; i++) {
        p.set("style", style.data());
        p.compute(c->be, c->err);
        ggml_tensor *t = p["img"];
        ggml_backend_tensor_get(t, img.data(), 0, ggml_nbytes(t));
    }
    printf("full decode: %.3f ms\n", (ggml_time_us()-t0)/1000.0/it);
    return 0;
}
