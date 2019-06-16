#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

/**
 * sum up small step of area to get approx integral result
 * integral 4.0 / ( 1 + x^2) dx from 0 to 1, get pi
 */

constexpr int num_steps = 1000000;
constexpr double step = 1.0 / num_steps;

double serial_version() {
    double sum = 0;
    for (int i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    double result = step * sum;
    return result;
}

double parallel_for_version() {
    double sum = 0;

    #pragma omp parallel for reduction(+ : sum) num_threads(4)
    for (int i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double result = step * sum;
    return result;
}

double parallel_version() {
    constexpr int thread_count = 4;
    vector<double> sums;
    sums.resize(4, 0);

    #pragma omp parallel num_threads(thread_count)
    {
        int id = omp_get_thread_num();
        // even you ask 4 threads, system may give you fewer threads, so check explicitly
        int actual_thread_count = omp_get_num_threads();
        const int thread_num_steps = num_steps / actual_thread_count;
        double sum = 0;  // avoid false sharing
        int offset = id * thread_num_steps;
        for (int i = 0; i < thread_num_steps; ++i) {
            double x = (i + offset + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        sums[id] = sum;
    }
    return std::accumulate(sums.begin(), sums.end(), 0.0) * step;
}

double parallel_critical_version() {
    constexpr int thread_count = 4;
    double result = 0;

    #pragma omp parallel num_threads(thread_count)
    {
        int id = omp_get_thread_num();
        // even you ask 4 threads, system may give you fewer threads, so check explicitly
        int actual_thread_count = omp_get_num_threads();
        const int thread_num_steps = num_steps / actual_thread_count;
        double sum = 0;  // avoid false sharing
        int offset = id * thread_num_steps;
        for (int i = 0; i < thread_num_steps; ++i) {
            double x = (i + offset + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp critical
        { result += sum; }
    }
    return result * step;
}

int main() {
    omp_set_num_threads(4);

    double t1 = omp_get_wtime();
    double result = serial_version();
    cout << "serial_version " << result << " took: " << (omp_get_wtime() - t1) << endl;

    t1 = omp_get_wtime();
    result = parallel_for_version();
    cout << "parallel_for_version " << result << " took: " << (omp_get_wtime() - t1) << endl;

    t1 = omp_get_wtime();
    result = parallel_version();
    cout << "parallel_version " << result << " took: " << (omp_get_wtime() - t1) << endl;

    t1 = omp_get_wtime();
    result = parallel_critical_version();
    cout << "parallel_critical_version " << result << " took: " << (omp_get_wtime() - t1) << endl;
    return 0;
}
