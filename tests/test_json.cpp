#undef NDEBUG  // asserts are the test
#include "json_min.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace tp;

int main() {
    jval v;
    std::string err;
    // basic parse
    assert(json_parse("{\"a\": [1, 2.5, -3e2], \"b\": {\"c\": \"x\\ny\"}, "
                      "\"t\": true, \"n\": null}", v, err));
    assert(v.kind == jval::OBJ);
    assert(v.find("a") && v.find("a")->arr.size() == 3);
    assert(v.find("a")->arr[1].num == 2.5);
    assert(v.find("b")->find("c")->str == "x\ny");
    // rejects
    jval w;
    err.clear();
    assert(!json_parse("{", w, err));
    err.clear();
    assert(!json_parse("[1,]", w, err));
    err.clear();
    assert(!json_parse("{\"a\":1} x", w, err));
    err.clear();
    // depth bomb must not crash
    std::string bomb(1000, '[');
    assert(!json_parse(bomb, w, err));
    // roundtrip
    std::string out = json_write(v, 1);
    jval v2;
    err.clear();
    assert(json_parse(out, v2, err));
    assert(v2.find("a")->arr.size() == 3);
    printf("test_json OK\n");
    return 0;
}
