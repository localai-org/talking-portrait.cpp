// meta.json / dump-parser fuzz (json_min + fixture meta interpretation)
#include "json_min.h"

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    std::string text((const char *) data, size);
    tp::jval v;
    std::string err;
    if (tp::json_parse(text, v, err)) {
        // exercise typical meta.json accessors
        (void) v.num_or("n_frames", 0);
        (void) v.str_or("case", "");
        const tp::jval *taps = v.find("taps");
        if (taps && taps->kind == tp::jval::OBJ) {
            for (auto &kv : taps->obj) {
                const tp::jval *sh = kv.second.find("shape");
                if (sh && sh->kind == tp::jval::ARR) {
                    long long n = 1;
                    for (auto &d : sh->arr) n *= (long long) d.num;
                    (void) n;
                }
            }
        }
        // roundtrip
        std::string out = tp::json_write(v, 1);
        tp::jval v2;
        err.clear();
        tp::json_parse(out, v2, err);
    }
    return 0;
}
