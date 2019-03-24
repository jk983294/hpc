#include <hash/dense_hash_map.h>
#include <catch.hpp>
#include <string>

using namespace flux;
using namespace std;

TEST_CASE("hash map", "[DenseHashMap]") {
    flux::dense_hash_map<string, int> m;
    m.set_empty_key("");
    m.set_deleted_key(" ");
    REQUIRE(m.find("") == m.end());

    for (int i = 0; i < 10; ++i) {
        m.insert({std::to_string(i), i});
    }

    for (int i = 0; i < 10; ++i) {
        string key = std::to_string(i);
        REQUIRE(m[key] == i);
    }

    for (int i = 0; i < 10; ++i) {
        m.erase({std::to_string(i)});
    }

    REQUIRE(m.empty());
}

TEST_CASE("hash set", "[DenseHashMap]") {
    flux::dense_hash_set<string> s;
    s.set_empty_key("");
    s.set_deleted_key(" ");
    REQUIRE(s.find("") == s.end());

    for (int i = 0; i < 10; ++i) {
        s.insert(std::to_string(i));
    }

    for (int i = 0; i < 10; ++i) {
        string key = std::to_string(i);
        REQUIRE(s.find(key) != s.end());
    }

    for (int i = 0; i < 10; ++i) {
        s.erase({std::to_string(i)});
    }

    REQUIRE(s.empty());
}
