#include <omp.h>
#include <utils/BenchHelper.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int len = 1000000000;
constexpr int bin_count = len / 10;
constexpr int thread_count = 4;

struct Counter {
    int64_t count{0};
    int64_t pad[7]; // padding to avoid false sharing, cache line is 64 bytes

    Counter() = default;
    void clear() { count = 0; }
};

void serial_version(const vector<double>& data, vector<Counter>& histogram) {
    for (int i = 0; i < len; ++i) {
        int bin_index = static_cast<int>(data[i] / 10);
        ++histogram[bin_index].count;
    }
}

void parallel_version(const vector<double>& data, vector<Counter>& histogram) {
    vector<omp_lock_t> locks(bin_count);

    for (int i = 0; i < bin_count; ++i) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int i = 0; i < len; ++i) {
        int bin_index = static_cast<int>(data[i] / 10);

        /**
         * the idea is that since we have huge bin numbers, so the collision chance is small
         * use a lock to mutual exclusively update histogram
         */
        omp_set_lock(&locks[bin_index]);
        ++histogram[bin_index].count;  // padding to avoid false sharing
        omp_unset_lock(&locks[bin_index]);
    }

    for (int i = 0; i < bin_count; ++i) {
        omp_destroy_lock(&locks[i]);
    }
}

int main() {
    static_assert(sizeof(Counter) == 64, "Counter is not 64 bytes aligned");
    omp_set_num_threads(4);

    vector<double> data = flux::GenRandomNumbers(0.0, len * 1.0, len);
    vector<Counter> histogram(bin_count);

    double t1 = omp_get_wtime();
    serial_version(data, histogram);
    cout << "serial_version " << histogram.front().count << " " << histogram.back().count << " took: " << (omp_get_wtime() - t1)
         << endl;

    for (auto& counter : histogram) {
        counter.clear();
    }
    t1 = omp_get_wtime();
    parallel_version(data, histogram);
    cout << "parallel_version " << histogram.front().count << " " << histogram.back().count << " took: " << (omp_get_wtime() - t1)
         << endl;

    cout << "done" << endl;
    return 0;
}
