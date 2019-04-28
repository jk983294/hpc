#include <benchmark/benchmark.h>
#include <vector>
#include <random>

using namespace std;

//$ cat /sys/devices/system/cpu/cpu0/cache/index1/size
//32K
//$ cat /sys/devices/system/cpu/cpu0/cache/index2/size
//1024K
//$ cat /sys/devices/system/cpu/cpu0/cache/index3/size
//25344K

//-------------------------------------------------------------------------
//Benchmark               Time             CPU   Iterations UserCounters...
//-------------------------------------------------------------------------
//bench_cache/13        421 ns          420 ns      1665569 bytes_per_second=18.1564G/s 8kb
//bench_cache/14        837 ns          835 ns       838858 bytes_per_second=18.2763G/s 16kb
//bench_cache/15       1671 ns         1666 ns       420965 bytes_per_second=18.3208G/s 32kb    // L1
//bench_cache/16       3444 ns         3435 ns       203572 bytes_per_second=17.7679G/s 64kb
//bench_cache/17       8207 ns         8184 ns        86043 bytes_per_second=14.9165G/s 128kb
//bench_cache/18      18671 ns        18617 ns        37708 bytes_per_second=13.1138G/s 256kb
//bench_cache/19      40028 ns        39912 ns        17505 bytes_per_second=12.234G/s 512kb
//bench_cache/20     103232 ns       102929 ns         6777 bytes_per_second=9.48775G/s 1024kb  // L2
//bench_cache/21     297896 ns       297029 ns         2358 bytes_per_second=6.57553G/s 2048kb
//bench_cache/22     884459 ns       881891 ns          790 bytes_per_second=4.4294G/s 4096kb
//bench_cache/23    2207082 ns      2200517 ns          318 bytes_per_second=3.5503G/s 8192kb
//bench_cache/24    5294615 ns      5278359 ns          126 bytes_per_second=2.9602G/s 16384kb
//bench_cache/25   16716210 ns     16662659 ns           41 bytes_per_second=1.87545G/s 32768kb
//bench_cache/26   54724004 ns     54571024 ns           12 bytes_per_second=1.1453G/s 65536kb

/**
 * ./cache_bench --benchmark_filter=bench_cache/16 --benchmark_min_time=2
 * perf stat ./cache_bench --benchmark_filter=bench_cache/15 --benchmark_min_time=2
 * check cache miss rate as follow:
 * perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-icache-load-misses,L1-icache-loads ./cache_bench --benchmark_filter=bench_cache/15 --benchmark_min_time=2
 */

void gen_random_data(vector<int>& v, int count, int lowBound, int upBound) {
    random_device rd;
    mt19937 generator(rd());
    std::uniform_int_distribution<int> uid(lowBound, upBound - 1); // [lowBound, upBound)
    for (int i = 0; i < count; ++i) {
        v.push_back(uid(generator));
    }
}

void bench_cache(benchmark::State& state) {
    int bytes = 1 << state.range(0);
    int count = (bytes / sizeof(int)) / 2;
    vector<int> v;
    gen_random_data(v, count, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    vector<int> indices;
    gen_random_data(indices, count, 0, count);
    for (auto _ : state) {
        long sum = 0;
        for(int i : indices) {
            sum += v[i];    // random retrieve data
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(long(state.iterations()) * long(bytes));
    state.SetLabel(std::to_string(bytes / 1024) + "kb");
}

BENCHMARK(bench_cache)->DenseRange(13, 26)->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
