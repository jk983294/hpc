#include <benchmark/benchmark.h>
#include <sys/time.h>
#include <chrono>

using namespace std;

//$ taskset -c 3 ./time_bench
// Run on (16 X 3200 MHz CPU s)
// CPU Caches:
//  L1 Data 32K (x16)
//  L1 Instruction 32K (x16)
//  L2 Unified 1024K (x16)
//  L3 Unified 25344K (x2)
// Load Average: 6.06, 6.08, 6.28
//-------------------------------------------------------------------------------
// Benchmark                                     Time             CPU   Iterations
//-------------------------------------------------------------------------------
// bench_timespec_CLOCK_REALTIME              27.9 ns         27.9 ns     25107361
// bench_timespec_CLOCK_REALTIME_COARSE       5.34 ns         5.32 ns    131476354   # !!! only ms precision
// bench_timeval                              31.0 ns         30.9 ns     22628471
// bench_rdtsc                                7.87 ns         7.85 ns     89165527

void bench_timespec_CLOCK_REALTIME(benchmark::State& state) {
    for (auto _ : state) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        benchmark::DoNotOptimize(ts.tv_sec * 1e9 + ts.tv_nsec);
    }
}

BENCHMARK(bench_timespec_CLOCK_REALTIME);

void bench_timespec_CLOCK_REALTIME_COARSE(benchmark::State& state) {
    for (auto _ : state) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME_COARSE, &ts);
        benchmark::DoNotOptimize(ts.tv_sec * 1e9 + ts.tv_nsec);
    }
}

BENCHMARK(bench_timespec_CLOCK_REALTIME_COARSE);

void bench_timeval(benchmark::State& state) {
    for (auto _ : state) {
        timeval tv;
        gettimeofday(&tv, nullptr);
        benchmark::DoNotOptimize(tv.tv_sec * 1e9 + tv.tv_usec * 1e3);
    }
}

BENCHMARK(bench_timeval);

inline uint64_t rdtsc() {
    unsigned long rax, rdx;
    asm volatile("rdtsc\n" : "=a"(rax), "=d"(rdx));
    return (rdx << 32) + rax;
}

void bench_rdtsc(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(rdtsc());
    }
}

BENCHMARK(bench_rdtsc);

BENCHMARK_MAIN();
