#include "json_min.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace tp {

namespace {

struct parser {
    const char *p, *end;
    std::string &err;
    int depth = 0;
    static const int MAX_DEPTH = 64;

    parser(const std::string &s, std::string &e) : p(s.data()), end(s.data() + s.size()), err(e) {}

    bool fail(const char *m) {
        if (err.empty()) err = m;
        return false;
    }
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    bool lit(const char *s, size_t n) {
        if ((size_t) (end - p) < n || memcmp(p, s, n) != 0) return false;
        p += n;
        return true;
    }
    bool parse_string(std::string &out) {
        if (p >= end || *p != '"') return fail("expected string");
        p++;
        out.clear();
        while (p < end && *p != '"') {
            unsigned char c = (unsigned char) *p;
            if (c == '\\') {
                if (p + 1 >= end) return fail("bad escape");
                p++;
                switch (*p) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (end - p < 5) return fail("bad \\u escape");
                        unsigned v = 0;
                        for (int i = 1; i <= 4; i++) {
                            char h = p[i];
                            v <<= 4;
                            if (h >= '0' && h <= '9') v |= (unsigned) (h - '0');
                            else if (h >= 'a' && h <= 'f') v |= (unsigned) (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') v |= (unsigned) (h - 'A' + 10);
                            else return fail("bad hex in \\u");
                        }
                        p += 4;
                        // encode as UTF-8 (surrogates passed through as-is)
                        if (v < 0x80) out += (char) v;
                        else if (v < 0x800) {
                            out += (char) (0xC0 | (v >> 6));
                            out += (char) (0x80 | (v & 0x3F));
                        } else {
                            out += (char) (0xE0 | (v >> 12));
                            out += (char) (0x80 | ((v >> 6) & 0x3F));
                            out += (char) (0x80 | (v & 0x3F));
                        }
                        break;
                    }
                    default: return fail("bad escape char");
                }
                p++;
            } else if (c < 0x20) {
                return fail("control char in string");
            } else {
                out += (char) c;
                p++;
            }
            if (out.size() > (1u << 24)) return fail("string too long");
        }
        if (p >= end) return fail("unterminated string");
        p++;
        return true;
    }
    bool parse_value(jval &v) {
        if (++depth > MAX_DEPTH) return fail("depth limit");
        ws();
        if (p >= end) return fail("unexpected end");
        bool ok = false;
        switch (*p) {
            case '{': {
                p++;
                v.kind = jval::OBJ;
                ws();
                if (p < end && *p == '}') {
                    p++;
                    ok = true;
                    break;
                }
                for (;;) {
                    ws();
                    std::string key;
                    if (!parse_string(key)) return false;
                    ws();
                    if (p >= end || *p != ':') return fail("expected ':'");
                    p++;
                    v.obj.emplace_back(std::move(key), jval{});
                    if (!parse_value(v.obj.back().second)) return false;
                    ws();
                    if (p < end && *p == ',') {
                        p++;
                        continue;
                    }
                    if (p < end && *p == '}') {
                        p++;
                        ok = true;
                        break;
                    }
                    return fail("expected ',' or '}'");
                }
                break;
            }
            case '[': {
                p++;
                v.kind = jval::ARR;
                ws();
                if (p < end && *p == ']') {
                    p++;
                    ok = true;
                    break;
                }
                for (;;) {
                    v.arr.emplace_back();
                    if (!parse_value(v.arr.back())) return false;
                    ws();
                    if (p < end && *p == ',') {
                        p++;
                        continue;
                    }
                    if (p < end && *p == ']') {
                        p++;
                        ok = true;
                        break;
                    }
                    return fail("expected ',' or ']'");
                }
                break;
            }
            case '"':
                v.kind = jval::STR;
                ok = parse_string(v.str);
                break;
            case 't':
                v.kind = jval::BOOL;
                v.b = true;
                ok = lit("true", 4) || fail("bad literal");
                break;
            case 'f':
                v.kind = jval::BOOL;
                v.b = false;
                ok = lit("false", 5) || fail("bad literal");
                break;
            case 'n':
                v.kind = jval::NUL;
                ok = lit("null", 4) || fail("bad literal");
                break;
            default: {
                char *num_end = nullptr;
                // strtod is fine here: the slice is NUL-terminated only at
                // the very end of the backing string, so bound-check first
                std::string tok;
                const char *q = p;
                while (q < end && (isdigit((unsigned char) *q) || *q == '-' || *q == '+' ||
                                   *q == '.' || *q == 'e' || *q == 'E'))
                    tok += *q++;
                if (tok.empty()) return fail("unexpected character");
                v.kind = jval::NUM;
                v.num = strtod(tok.c_str(), &num_end);
                if (num_end != tok.c_str() + tok.size()) return fail("bad number");
                if (!std::isfinite(v.num)) return fail("non-finite number");
                p = q;
                ok = true;
                break;
            }
        }
        depth--;
        return ok;
    }
};

void write_str(std::string &o, const std::string &s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    o += '"';
}

void write_val(std::string &o, const jval &v, int indent, int level) {
    const std::string pad(indent * (level + 1), ' ');
    const std::string pad0(indent * level, ' ');
    const char *nl = indent ? "\n" : "";
    switch (v.kind) {
        case jval::NUL: o += "null"; break;
        case jval::BOOL: o += v.b ? "true" : "false"; break;
        case jval::NUM: {
            char buf[64];
            if (std::fabs(v.num) < 1e15 && v.num == (double) (int64_t) v.num)
                snprintf(buf, sizeof buf, "%lld", (long long) v.num);
            else
                snprintf(buf, sizeof buf, "%.9g", v.num);
            o += buf;
            break;
        }
        case jval::STR: write_str(o, v.str); break;
        case jval::ARR: {
            if (v.arr.empty()) {
                o += "[]";
                break;
            }
            o += "[";
            for (size_t i = 0; i < v.arr.size(); i++) {
                o += nl;
                if (indent) o += pad;
                write_val(o, v.arr[i], indent, level + 1);
                if (i + 1 < v.arr.size()) o += ",";
            }
            o += nl;
            if (indent) o += pad0;
            o += "]";
            break;
        }
        case jval::OBJ: {
            if (v.obj.empty()) {
                o += "{}";
                break;
            }
            o += "{";
            for (size_t i = 0; i < v.obj.size(); i++) {
                o += nl;
                if (indent) o += pad;
                write_str(o, v.obj[i].first);
                o += indent ? ": " : ":";
                write_val(o, v.obj[i].second, indent, level + 1);
                if (i + 1 < v.obj.size()) o += ",";
            }
            o += nl;
            if (indent) o += pad0;
            o += "}";
            break;
        }
    }
}

}  // namespace

bool json_parse(const std::string &text, jval &out, std::string &err) {
    if (text.size() > (1u << 28)) {
        err = "input too large";
        return false;
    }
    parser ps(text, err);
    if (!ps.parse_value(out)) return false;
    ps.ws();
    if (ps.p != ps.end) {
        err = "trailing garbage";
        return false;
    }
    return true;
}

std::string json_write(const jval &v, int indent) {
    std::string o;
    write_val(o, v, indent, 0);
    if (indent) o += "\n";
    return o;
}

}  // namespace tp
