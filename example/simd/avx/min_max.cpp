#include <immintrin.h>
#include <iostream>

using namespace std;

/**
 * g++ -std=c++11 -mavx min_max.cpp
 * cond ? a : b
 * first evaluate both sides, giving a and b. then evaluate the condition using a SIMD compare,
 * which returns a vector containing a bit mask that is has all bits set for lanes that meet cond
 */

int main() {
    __m256 a = _mm256_setr_ps(1.0, 6.0, 9.0, 4.0, 2.0, 6.0, 1.0, 8.0);
    __m256 b = _mm256_setr_ps(1.0, 3.0, 3.0, 12.0, 5.0, 7.0, 9.0, 6.0);

    __m256 mask = __m256 _mm256_cmp_ps(a, b, _CMP_GE_OS);
    __m256 minValues = _mm256_blendv_ps(a, b, mask);

    float* f = (float*)&minValues;
    printf("min: %f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

    mask = __m256 _mm256_cmp_ps(a, b, _CMP_LE_OS);
    __m256 maxValues = _mm256_blendv_ps(a, b, mask);

    f = (float*)&maxValues;
    printf("max: %f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    return 0;
}
