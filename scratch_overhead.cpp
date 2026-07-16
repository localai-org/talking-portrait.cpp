#undef NDEBUG
#include "tp_internal.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace tp;
int main(int argc, char **argv) {
    std::string err;
    backend_t be; assert(be.init("gpu", 0, err));
    for (int n_ops : {10, 100, 400}) {
        plan_t p; p.begin(n_ops + 8);
        ggml_tensor *x = p.input("x", ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, 256));
        ggml_tensor *t = x;
        for (int i = 0; i < n_ops; i++) t = ggml_scale(p.ctx, t, 1.0001f);
        p.output("y", t);
        assert(p.commit(be, err));
        std::vector<float> xv(256, 1.0f);
        p.set("x", xv.data());
        for (int i = 0; i < 5; i++) p.compute(be, err);
        ggml_backend_synchronize(be.be);
        int64_t t0 = ggml_time_us();
        const int it = 30;
        for (int i = 0; i < it; i++) p.compute(be, err);
        ggml_backend_synchronize(be.be);
        double ms = (ggml_time_us() - t0) / 1000.0 / it;
        printf("n_ops=%4d: %8.3f ms  (%.1f us/op)\n", n_ops, ms, ms * 1000.0 / n_ops);
    }
    // one medium GEMM for reference
    {
        plan_t p; p.begin(16);
        ggml_tensor *a = p.input("a", ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 576, 32));
        ggml_tensor *b = p.input("b", ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 576, 262144));
        p.output("y", ggml_mul_mat(p.ctx, a, b));
        assert(p.commit(be, err));
        for (int i = 0; i < 3; i++) p.compute(be, err);
        ggml_backend_synchronize(be.be);
        int64_t t0 = ggml_time_us();
        for (int i = 0; i < 20; i++) p.compute(be, err);
        ggml_backend_synchronize(be.be);
        printf("gemm 32x576x262144: %.3f ms\n", (ggml_time_us() - t0) / 1000.0 / 20);
    }
    return 0;
}
