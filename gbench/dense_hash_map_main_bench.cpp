#include <benchmark/benchmark.h>
#include <hash/dense_hash_map.h>
#include <utils/BenchHelper.h>
#include <zerg_string.h>
#include <algorithm>
#include <random>
#include <unordered_map>

using namespace std;

class MyFixture : public benchmark::Fixture {
public:
    MyFixture() {
        dense_map.set_empty_key("");
        dense_map.set_deleted_key(" ");
        all_keys = flux::GetRandomStringFromFile();
    }

    void SetUp(const ::benchmark::State& state) {
        int64_t count = state.range(0);
        test_count = std::min(test_count, count);
        for (int64_t i = 0; i < count; ++i) {
            string randStr = all_keys[i];
            key_can_found.push_back(randStr);
            std_map.insert({randStr, randStr});
            dense_map.insert({randStr, randStr});
        }

        for (int64_t i = 0; i < test_count; ++i) {
            string randStr = all_keys[i + count];
            key_can_not_found.push_back(randStr);
        }

        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(key_can_found), std::end(key_can_found), rng);
        std::copy(std::begin(key_can_found), std::begin(key_can_found) + test_count, std::back_inserter(key2insert));
        key_can_found.resize(test_count);  // only keep test_count elements
    }

    void TearDown(const ::benchmark::State& state) {
        std_map.clear();
        dense_map.clear();
        key_can_found.clear();
        key_can_not_found.clear();
    }

    template <typename TMap>
    int64_t find_test(const TMap& m, const vector<string>& keys) {
        int64_t count = 0;
        for (const auto& str : keys) {
            if (m.find(str) != m.end()) {
                ++count;
            }
        }
        return count;
    }

    template <typename TMap>
    size_t insert_test(TMap& m) {
        m.clear();
        for (const auto& str : key2insert) {
            m.insert({str, str});
        }
        return m.size();
    }

    int64_t test_count{1000};
    vector<string> all_keys;
    vector<string> key_can_found;
    vector<string> key_can_not_found;
    vector<string> key2insert;
    std::unordered_map<string, string> std_map;
    flux::dense_hash_map<string, string> dense_map;
};

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, key_can_found));
    }
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_find)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14);

BENCHMARK_DEFINE_F(MyFixture, dense_hash_map_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(dense_map, key_can_found));
    }
}

BENCHMARK_REGISTER_F(MyFixture, dense_hash_map_find)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(std_map, key_can_not_found));
    }
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_not_find)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14);

BENCHMARK_DEFINE_F(MyFixture, dense_hash_map_not_find)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(find_test(dense_map, key_can_not_found));
    }
}

BENCHMARK_REGISTER_F(MyFixture, dense_hash_map_not_find)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14);

BENCHMARK_DEFINE_F(MyFixture, std_hash_map_insert)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(insert_test(std_map));
    }
}

BENCHMARK_REGISTER_F(MyFixture, std_hash_map_insert)->Arg(1 << 10);

BENCHMARK_DEFINE_F(MyFixture, dense_hash_map_insert)(benchmark::State& st) {
    for (auto _ : st) {
        benchmark::DoNotOptimize(insert_test(dense_map));
    }
}

BENCHMARK_REGISTER_F(MyFixture, dense_hash_map_insert)->Arg(1 << 10);

int main(int argc, char** argv) {
    flux::GenRandomStringFile();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
