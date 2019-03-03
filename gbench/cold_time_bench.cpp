#include <benchmark/benchmark.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

using namespace std;

inline uint64_t rdtsc() {
    unsigned long rax, rdx;
    asm volatile("rdtsc\n" : "=a"(rax), "=d"(rdx));
    return (rdx << 32) + rax;
}

int main() {
    struct timespec ts;
    while (1) {
        /**
         * first warm up call is crucial
         * if no this warm up call, you will probably get 500 ns because one instruction and data tlb miss,
         * each tlb miss cause around 200ns
         */
        clock_gettime(CLOCK_REALTIME, &ts);

        /**
         * only measure second call
         */
        auto t1 = rdtsc();
        clock_gettime(CLOCK_REALTIME, &ts);
        benchmark::DoNotOptimize(ts.tv_sec * 1e9 + ts.tv_nsec);
        auto t2 = rdtsc();
        cout << (t2 - t1) << " " << ((t2 - t1) / 2.3) << " ns" << endl;  // 2.3 from benchmark machine 2.3GHz
        sleep(1);
    }
}
