#include <immintrin.h>
#include <iostream>

using namespace std;

/**
 * g++ -std=c++11 -mavx sub.cpp
 */

int main() {
    // initialize the two argument vectors
    __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    __m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);

    // compute the difference between the two vectors
    __m256 result = _mm256_sub_ps(evens, odds);

    // display the elements of the result vector
    float* f = (float*)&result;
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    return 0;
}
