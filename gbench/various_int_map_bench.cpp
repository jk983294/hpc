#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include "FlatMap.h"

using namespace std;

using FlatMapII = FlatMap<int, int>;

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() {}

    ~MyFixture() {}

    void SetUp(const ::benchmark::State& state) {
        int64_t stock_count = state.range(0);
        for (int i = 0; i < static_cast<int>(stock_count); ++i) {
            std_map[i] = i;
            flat_map.insert({i, i});

            access_sequence_all.push_back(i);
            not_found_access_sequence_all.push_back(i + stock_count);
        }

        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(access_sequence_all), std::end(access_sequence_all), rng);
        std::shuffle(std::begin(not_found_access_sequence_all), std::end(not_found_access_sequence_all), rng);

        access_sequence.resize(16);
        not_found_access_sequence.resize(16);

        std::copy(access_sequence_all.begin(), access_sequence_all.begin() + 16, access_sequence.begin());
        std::copy(not_found_access_sequence_all.begin(), not_found_access_sequence_all.begin() + 16,
                  not_found_access_sequence.begin());
    }

    void TearDown(const ::benchmark::State& state) {
        std_map.clear();
        flat_map.clear();
        access_sequence.clear();
    }

    template <typename TMap, typename TVector>
    int64_t find_test(const TMap& m, TVector& v) {
        int64_t count = 0;
        for (int offset : v) {
            if (m.find(offset) != m.end()) {
                ++count;
            }
        }
        return count;
    }

    std::unordered_map<int, int> std_map;
    FlatMapII flat_map;
    vector<int> access_sequence_all;
    vector<int> not_found_access_sequence_all;

    vector<int> access_sequence;
    vector<int> not_found_access_sequence;
};

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, access_sequence));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_find)->Arg(1 << 4)->Arg(1 << 5)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, flat_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(flat_map, access_sequence));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, flat_map_find)->Arg(1 << 4)->Arg(1 << 5)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, not_found_access_sequence));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_not_find)
    ->Arg(1 << 4)
    ->Arg(1 << 5)
    ->Arg(1 << 10)
    ->Arg(1 << 16)
    ->Arg(1 << 20);

BENCHMARK_DEFINE_F(MyFixture, flat_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(flat_map, not_found_access_sequence));
    }
    st.SetItemsProcessed(st.iterations() * 16);
}

BENCHMARK_REGISTER_F(MyFixture, flat_map_not_find)->Arg(1 << 4)->Arg(1 << 5)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20);

BENCHMARK_MAIN();
