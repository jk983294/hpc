#include <benchmark/benchmark.h>
#include <sys/time.h>
#include <cmath>

using namespace std;

//--------------------------------------------------------
// Benchmark              Time             CPU   Iterations
//--------------------------------------------------------
// bench_pow_1_4      0.314 ns        0.313 ns   1000000000
// bench_pow_1_5      0.314 ns        0.313 ns   1000000000

void bench_pow_1_4(benchmark::State& state) {
    auto base = 1.000000000001, exp = 1.4;
    for (auto _ : state) {
        auto result = std::pow(base, exp);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(bench_pow_1_4);

void bench_pow_1_5(benchmark::State& state) {
    auto base = 1.000000000001, exp = 1.5;
    for (auto _ : state) {
        auto result = std::pow(base, exp);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(bench_pow_1_5);

BENCHMARK_MAIN();
