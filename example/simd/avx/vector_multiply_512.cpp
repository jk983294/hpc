#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

/**
 * g++ -std=c++11 -march=native vector_multiply_512.cpp
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

static inline double horizontal_add (__m256d a) {
    __m256d t1 = _mm256_hadd_pd(a,a);
    __m128d t2 = _mm256_extractf128_pd(t1,1);
    __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1),t2);
    return _mm_cvtsd_f64(t3);
}

double simd_vec_multiply(double* x, double* y, int n) {
    constexpr int xmmBits = 512;
    constexpr int xmmAlignBytes = xmmBits / 8;
    constexpr int elementCountPerAlign = xmmAlignBytes / sizeof(double);
    __m512d msum1 = _mm512_setzero_pd();

    for (int i = 0; i < n; i += elementCountPerAlign) {
        __m512d vec_x = _mm512_loadu_pd(x + i);
        __m512d vec_y = _mm512_loadu_pd(y + i);
        msum1 = _mm512_fmadd_pd(vec_x, vec_y, msum1);
    }
    return _mm512_reduce_add_pd(msum1);
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
