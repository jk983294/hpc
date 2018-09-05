#include <immintrin.h>
#include <iostream>

using namespace std;

/**
 * g++ -std=c++11 -mavx branch.cpp
 * cond ? a : b
 * first evaluate both sides, giving a and b. then evaluate the condition using a SIMD compare,
 * which returns a vector containing a bit mask that is has all bits set for lanes that meet cond
 */

int main() {
    __m256 a = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    __m256 b = _mm256_setr_ps(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f);

    __m256 c = _mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 comparator = _mm256_set1_ps(4);
    __m256 mask = __m256 _mm256_cmp_ps(c, comparator, _CMP_GE_OS);
    __m256 result1 = _mm256_blendv_ps(a, b, mask);

    // display the elements of the result vector
    float* f = (float*)&result1;
    printf("%f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

    float e[8];
    for (int i = 0; i < 8; ++i) {
        if (i >= 4) {
            e[i] = i + 1;
        } else {
            e[i] = -(i + 1);
        }
    }
    printf("%f %f %f %f %f %f %f %f\n", e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
    return 0;
}
