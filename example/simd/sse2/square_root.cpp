#include <emmintrin.h>
#include <cmath>
#include "utils/Timer.h"

void normal(float* a, int N) {
    for (int i = 0; i < N; ++i) a[i] = std::sqrt(a[i]);
}

void sse(float* a, int N) {
    // We assume N % 4 == 0
    int itrCount = N / 4;
    __m128* ptr = (__m128*)a;

    for (int i = 0; i < itrCount; ++i, ++ptr, a += 4) {
        _mm_store_ps(a, _mm_sqrt_ps(*ptr));
    }
}

int main() {
    int N = 1024 * 1024 * 64;

    float* a;
    /**
     * posix_memalign allocates aligned data on the heap
     * gcc attribute allocates aligned data on the stack
     */
    posix_memalign((void**)&a, 16, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = 3141592.65358;
    }

    {
        TIMER("normal");
        normal(a, N);
        cout << "normal result: " << a[0] << endl;
    }

    for (int i = 0; i < N; ++i) {
        a[i] = 3141592.65358;
    }

    {
        TIMER("sse");
        sse(a, N);
        cout << "sse result: " << a[0] << endl;
    }
    return 0;
}
