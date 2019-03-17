#include <dense_hash_map.h>
#include <benchmark/benchmark.h>
#include <zerg_string.h>
#include <unordered_map>

using namespace std;

void bench_GenerateRandomString(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(ztool::GenerateRandomString(8));
    }
}

BENCHMARK(bench_GenerateRandomString);

static std::unordered_map<string, string> generate_std_map(int count) {
    std::unordered_map<string, string> m;
    for (int i = 0; i < count; ++i) {
        string randStr = ztool::GenerateRandomString(8);
        m.insert({randStr, randStr});
    }
    return m;
}

void bench_std_hash_map_find(benchmark::State& state) {
    std::unordered_map<string, string> m = generate_std_map(state.range(0));
    for (auto _ : state) {
        string randStr = ztool::GenerateRandomString(8);
        bool found = m.find(randStr) == m.end();
        benchmark::DoNotOptimize(found);
    }
}

BENCHMARK(bench_std_hash_map_find)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14);

void bench_dense_hash_map_find(benchmark::State& state) {
    flux::dense_hash_map<string, string> m;
    m.set_empty_key("");
    m.set_deleted_key(" ");
    for (int i = 0; i < state.range(0); ++i) {
        string randStr = ztool::GenerateRandomString(8);
        if (randStr.empty() || randStr == " ") continue;
        m.insert({randStr, randStr});
    }

    for (auto _ : state) {
        string randStr = ztool::GenerateRandomString(8);
        bool found = m.find(randStr) == m.end();
        benchmark::DoNotOptimize(found);
    }
}

BENCHMARK(bench_dense_hash_map_find)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14);

BENCHMARK_MAIN();
