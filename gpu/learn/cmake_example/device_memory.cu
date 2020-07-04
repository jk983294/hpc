#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_one(int n, float* x) {
    int i = threadIdx.x;
    if (i < n) {
        x[i] = x[i] + 1;
        printf("thread %d, value=%f\n", i, x[i]);
    }
}

void initialize_input(float* h_A, int n) {
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
    }
}

__constant__ float constData[256];
void test1(int n, float* x) {
    float data[256];
    cudaMemcpyToSymbol(constData, data, sizeof(data));
    cudaMemcpyFromSymbol(data, constData, sizeof(data));
}

__device__ float devData;
void test2(int n, float* x) {
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
}

__device__ float* devPointer;
void test3(int n, float* x) {
    float* ptr;
    cudaMalloc(&ptr, 256 * sizeof(float));
    cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
}

int main(void) {
    int N = 16;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A in host memory
    float* h_A = (float*)malloc(size);

    initialize_input(h_A, N);

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    add_one<<<1, N>>>(N, d_A);

    // Copy result from device memory to host memory
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);

    printf("result: %f,%f\n", h_A[0], h_A[1]);

    // Free host memory
    free(h_A);
}
