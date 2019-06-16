#include <omp.h>
#include <utils/BenchHelper.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int len = 10000000;
constexpr int bin_count = len / 10;
constexpr int thread_count = 4;

void serial_version(const vector<double>& data, vector<int>& histogram) {
    for (int i = 0; i < len; ++i) {
        int bin_index = static_cast<int>(data[i] / 10);
        ++histogram[bin_index];
    }
}

void parallel_version(const vector<double>& data, vector<int>& histogram) {
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
        ++histogram[bin_index];  // need padding to avoid false sharing
        omp_unset_lock(&locks[bin_index]);
    }

    for (int i = 0; i < bin_count; ++i) {
        omp_destroy_lock(&locks[i]);
    }
}

int main() {
    omp_set_num_threads(4);

    vector<double> data = flux::GenRandomNumbers(0.0, len * 1.0, len);
    vector<int> histogram(bin_count, 0);

    double t1 = omp_get_wtime();
    serial_version(data, histogram);
    cout << "serial_version " << histogram.front() << " " << histogram.back() << " took: " << (omp_get_wtime() - t1)
         << endl;

    std::fill(histogram.begin(), histogram.end(), 0);
    t1 = omp_get_wtime();
    parallel_version(data, histogram);
    cout << "parallel_version " << histogram.front() << " " << histogram.back() << " took: " << (omp_get_wtime() - t1)
         << endl;

    cout << "done" << endl;
    return 0;
}
