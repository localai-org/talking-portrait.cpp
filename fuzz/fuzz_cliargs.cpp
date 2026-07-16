// CLI arg-path fuzz: tokenize input into argv and drive the IO-free parser
#include "../tools/cli_args.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4096) return 0;
    std::string buf((const char *) data, size);
    std::vector<std::string> toks;
    toks.push_back("tp-cli");
    std::string cur;
    for (char c : buf) {
        if (c == ' ' || c == '\n' || c == '\0') {
            if (!cur.empty()) toks.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
        if (toks.size() > 64) break;
    }
    if (!cur.empty() && toks.size() <= 64) toks.push_back(cur);
    std::vector<const char *> argv;
    for (auto &t : toks) argv.push_back(t.c_str());
    tp::cli_opts o;
    std::string err;
    (void) tp::cli_parse((int) argv.size(), argv.data(), o, err);
    return 0;
}
