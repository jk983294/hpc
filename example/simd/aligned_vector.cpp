#include <iomanip>
#include <iostream>
#include <vector>
#include "simd/simd_aligned_allocator.h"

using namespace std;
using namespace flux;

int main() {
    typedef std::vector<__m128, SimdAlignedAllocator<__m128, sizeof(__m128)> > aligned_vector;
    aligned_vector lhs;
    aligned_vector rhs;

    float a = 1.0f;
    float b = 2.0f;
    float c = 3.0f;
    float d = 4.0f;

    float e = 5.0f;
    float f = 6.0f;
    float g = 7.0f;
    float h = 8.0f;

    for (std::size_t i = 0; i < 1000; ++i) {
        lhs.push_back(_mm_set_ps(a, b, c, d));
        rhs.push_back(_mm_set_ps(e, f, g, h));

        a += 1.0f;
        b += 1.0f;
        c += 1.0f;
        d += 1.0f;
        e += 1.0f;
        f += 1.0f;
        g += 1.0f;
        h += 1.0f;
    }

    __m128 mul = _mm_mul_ps(lhs[10], rhs[10]);

    float* addr = (float*)&(lhs[10]);
    printf("%f %f %f %f \n", addr[0], addr[1], addr[2], addr[3]);
    addr = (float*)&(rhs[10]);
    printf("%f %f %f %f \n", addr[0], addr[1], addr[2], addr[3]);
    addr = (float*)&(mul);
    printf("%f %f %f %f \n", addr[0], addr[1], addr[2], addr[3]);
    return 0;
}
