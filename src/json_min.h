// minimal JSON (parse + serialize) for harness meta.json interop.
// Untrusted-input surface: bounded depth, no exceptions escape parse().
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tp {

struct jval {
    enum kind_t { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<jval> arr;
    std::vector<std::pair<std::string, jval>> obj;  // preserves order

    const jval *find(const std::string &key) const {
        if (kind != OBJ) return nullptr;
        for (auto &kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    double num_or(const std::string &key, double dflt) const {
        const jval *v = find(key);
        return v && v->kind == NUM ? v->num : dflt;
    }
    std::string str_or(const std::string &key, const std::string &dflt) const {
        const jval *v = find(key);
        return v && v->kind == STR ? v->str : dflt;
    }
};

// returns false on malformed input; err gets a short reason
bool json_parse(const std::string &text, jval &out, std::string &err);

// serializer used for our meta.json (2-space indent, stable order)
std::string json_write(const jval &v, int indent = 0);

}  // namespace tp
