#include <immintrin.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

/**
 * g++ -std=c++11 -mavx vector_add.cpp
 */

int main() {
    constexpr int elementCount = 11111257;
    constexpr int xmmBits = 256;
    constexpr int xmmAlignBytes = xmmBits / 8;
    constexpr int elementCountPerAlign = xmmAlignBytes / sizeof(float);
    constexpr float step = 4.2;
    vector<float> v;
    for (int i = 0; i < elementCount; ++i) {
        v.push_back(i * 1.0f);
    }

    uint64_t addr = reinterpret_cast<uint64_t>(v.data());
    int offset = static_cast<int>(addr % xmmAlignBytes);
    int leadingElementCount = (elementCountPerAlign - (offset / sizeof(float))) % elementCountPerAlign;
    cout << addr << " " << offset << " " << leadingElementCount << endl;

    cout << "first " << leadingElementCount << " elements go to normal method due to alignment requirement" << endl;
    int i = 0;
    for (; i < leadingElementCount; ++i) {
        v[i] += step;
    }

    //__m256 increment = _mm256_set_ps(step, step, step, step, step, step, step, step);
    __m256 increment = _mm256_set1_ps(step);

    int len = v.size();
    for (i = leadingElementCount; i + elementCountPerAlign <= len; i += elementCountPerAlign) {
        /**
         * _mm256_loadu_ps/_mm256_storeu_ps does not need any align requirement, but give poor performance
         * _mm256_load_ps/_mm256_store_ps need aligned to 32-byte boundary which yields better performance
         */
        __m256 inner = _mm256_load_ps(v.data() + i);
        __m256 result = _mm256_add_ps(inner, increment);
        _mm256_store_ps(v.data() + i, result);
    }
    cout << "vectorized method handles " << (i - leadingElementCount) << " elements in middle" << endl;

    cout << "the last " << (v.size() - i) << " elements go back to un-vectorized method" << endl;
    for (int j = i; j < len; ++j) {
        v[j] += step;
    }

    cout << std::fixed << setprecision(2) << "result: " << v[0] << " " << v[len / 2] << " " << v[len - 1] << endl;
    return 0;
}
