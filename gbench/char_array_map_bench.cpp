#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

class StringHash {
public:
    unsigned long operator()(const char* Str) const {
        unsigned long Hash = 1315423911;
        while (*Str) {
            Hash ^= ((Hash << 5) + (*Str++) + (Hash >> 2));
        }
        return (Hash & 0x7FFFFFFF);
    }
};

class StringCmp {
public:
    bool operator()(const char* Val1, const char* Val2) const { return (strcmp(Val1, Val2) == 0); }
};

using CharArrayMap = std::unordered_map<const char*, int, StringHash, StringCmp>;

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() {}

    ~MyFixture() {}

    void SetUp(const ::benchmark::State& state) {
        int64_t stock_count = state.range(0);
        content = new char[stock_count * record_len];
        not_found_content = new char[stock_count * record_len];
        memset(content, 0, stock_count * record_len);
        memset(not_found_content, 0, stock_count * record_len);
        for (int i = 0; i < static_cast<int>(stock_count); ++i) {
            char* pStock = content + i * record_len;
            char* pNotFoundStock = not_found_content + i * record_len;
            snprintf(pStock, record_len, "%09d", i);
            snprintf(pNotFoundStock, record_len, "%09d,", i);

            std_map[pStock] = i;
            char_array_map[pStock] = i;

            access_sequence.push_back(i);

            content_pointers.push_back(pStock);
            not_found_content_pointers.push_back(pNotFoundStock);
            content_strings.push_back(pStock);
            not_found_content_strings.push_back(pNotFoundStock);
        }

        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(access_sequence), std::end(access_sequence), rng);
    }

    void TearDown(const ::benchmark::State& state) {
        if (content) delete[] content;
        if (not_found_content) delete[] not_found_content;
        std_map.clear();
        char_array_map.clear();
        access_sequence.clear();

        content_pointers.clear();
        not_found_content_pointers.clear();
        content_strings.clear();
        not_found_content_strings.clear();
    }

    template <typename TMap, typename TVector>
    int64_t find_test(const TMap& m, TVector& v) {
        int64_t count = 0;
        for (int offset : access_sequence) {
            // auto pStock = (content_ + offset * record_len);
            if (m.find(v[offset]) != m.end()) {
                ++count;
            }
        }
        return count;
    }

    template <typename TMap, typename TVector>
    int64_t insert_test(TMap& m, TVector& v) {
        int64_t count = 0;
        for (int offset : access_sequence) {
            // char* pStock = (content_ + offset * record_len);
            m[v[offset]] = offset;
            ++count;
        }
        return count;
    }

    template <typename TMap, typename TVector>
    int64_t erase_test(TMap& m, TVector& v) {
        int64_t count = 0;
        for (int offset : access_sequence) {
            // char* pStock = (content_ + offset * record_len);
            auto it = m.find(v[offset]);
            if (it != m.end()) {
                m.erase(it);
            }
            ++count;
        }
        return count;
    }

    //    size_t stock_count{3000};
    size_t record_len{12};
    char* content{nullptr};
    char* not_found_content{nullptr};
    std::unordered_map<string, int> std_map;
    CharArrayMap char_array_map;
    vector<int> access_sequence;

    vector<char*> content_pointers;
    vector<char*> not_found_content_pointers;
    vector<string> content_strings;
    vector<string> not_found_content_strings;
};

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_find_with_string_ctor)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_find_with_string_ctor)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, content_strings));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_find)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, char_array_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(char_array_map, content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, char_array_map_find)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_not_find_with_string_ctor)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, not_found_content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_not_find_with_string_ctor)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, not_found_content_strings));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_not_find)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, char_array_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(char_array_map, not_found_content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, char_array_map_not_find)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_insert)(benchmark::State& st) {
    std_map.clear();
    for (auto _ : st) {
        benchmark::DoNotOptimize(insert_test(std_map, content_strings));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_insert)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, char_array_map_insert)(benchmark::State& st) {
    char_array_map.clear();
    for (auto _ : st) {
        benchmark::DoNotOptimize(insert_test(char_array_map, content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, char_array_map_insert)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_erase)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(erase_test(std_map, content_strings));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_erase)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, char_array_map_erase)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(erase_test(char_array_map, content_pointers));
    }
    st.SetItemsProcessed(st.iterations() * access_sequence.size());
}

BENCHMARK_REGISTER_F(MyFixture, char_array_map_erase)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_MAIN();
