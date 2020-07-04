#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cstdio>
#include "a.cuh"

void test(int offset) {
    printf("start\n");

    thrust::device_vector<int> X(2 << offset);

    std::generate(X.begin(), X.end(), rand);
    int sum = thrust::reduce(X.begin(), X.end());
    printf("sum=%d\n", sum);

    printf("end\n");
}
