#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int len = 10000000;
constexpr int thread_count = 4;
constexpr int chunk_size = len / thread_count;

double serial_version(const vector<int>& v) {
    int result = 0;
    for (int i = 0; i < len; ++i) {
        result += v[i];
    }
    return result;
}

double parallel_version(const vector<int>& v) {
    int result = 0;

    #pragma omp parallel for reduction(+ : result) schedule(static, chunk_size) num_threads(thread_count)
    for (int i = 0; i < len; ++i) {
        result += v[i];
    }
    return result;
}

int main() {
    omp_set_num_threads(4);

    vector<int> a(len);
    std::iota(a.begin(), a.end(), -(len / 2));

    double t1 = omp_get_wtime();
    double result = serial_version(a);
    cout << "serial_version " << result << " took: " << (omp_get_wtime() - t1) << endl;

    t1 = omp_get_wtime();
    result = parallel_version(a);
    cout << "parallel_version " << result << " took: " << (omp_get_wtime() - t1) << endl;

    cout << "done " << endl;
    return 0;
}
