#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

/**
 * g++ -std=c++11 -mavx vector_multiply_256.cpp
 */

void genRandNums(double* x, int n, double a, double b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
    }
}

double plain_vec_multiply(double* x, double* y, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

double simd_vec_multiply(double* x, double* y, int n) {
    double res = 0;
    constexpr int xmmBits = 256;
    constexpr int xmmAlignBytes = xmmBits / 8;
    constexpr int elementCountPerAlign = xmmAlignBytes / sizeof(double);

    for (int i = 0; i < n; i += elementCountPerAlign) {
        __m256d vec_x = _mm256_loadu_pd(x + i);
        __m256d vec_y = _mm256_loadu_pd(y + i);
        auto vec_multi = _mm256_mul_pd(vec_x, vec_y);
        __m256d s = _mm256_hadd_pd(vec_multi, vec_multi);
        res += (((double*)&s)[0] + ((double*)&s)[2]);
    }
    return res;
}


int main() {
    int n = 4096;
    std::vector<double> a, b;
    a.resize(n, 0);
    b.resize(n, 0);
    genRandNums(a.data(), n, 0, 1);
    genRandNums(b.data(), n, 0, 1);

    cout << plain_vec_multiply(a.data(), b.data(), n) << endl;
    cout << simd_vec_multiply(a.data(), b.data(), n) << endl;

    int times = 10000;
    double res1 = 0, res2 = 0;
    int64_t ns1 = 0, ns2 = 0;
    for (int i = 0; i < times; ++i) {
        genRandNums(a.data(), n, 0, 1);
        genRandNums(b.data(), n, 0, 1);

        steady_clock::time_point t1 = steady_clock::now();
        res1 += plain_vec_multiply(a.data(), b.data(), n);
        steady_clock::time_point t2 = steady_clock::now();
        res2 += simd_vec_multiply(a.data(), b.data(), n);
        steady_clock::time_point t3 = steady_clock::now();

        ns1 += nanoseconds{t2 - t1}.count();
        ns2 += nanoseconds{t3 - t2}.count();
    }

    cout << ns1 << "," << (double)(ns1) / times << "," << res1 << endl;
    cout << ns2 << "," << (double)(ns2) / times << "," << res2 << endl;
}
