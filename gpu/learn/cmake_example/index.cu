#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_one_v1(int n, float* x) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    x[i + j * n] += 1;
}

__global__ void add_one_v2(int n, float* x) {
    int left_up_of_block_x = blockIdx.x * blockDim.x;
    int left_up_of_block_y = blockIdx.y * blockDim.y;
    int i = left_up_of_block_x + threadIdx.x;
    int j = left_up_of_block_y + threadIdx.y;
    if (i < n) {
        x[i + j * n] += 1;
    }
}

void initialize_input(float* h_A, int n) {
    for (int i = 0; i < n * n; i++) {
        h_A[i] = i;
    }
}

int main(void) {
    int N = 16;
    size_t size = N * N * sizeof(float);

    // Allocate input vectors h_A in host memory
    float* h_A = (float*)malloc(size);

    initialize_input(h_A, N);

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    add_one_v1<<<1, N>>>(N, d_A);

    // Copy result from device memory to host memory
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    printf("result: %f,%f,%f,%f\n", h_A[0], h_A[1], h_A[N * N - 2], h_A[N * N - 1]);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    add_one_v2<<<numBlocks, threadsPerBlock>>>(N, d_A);

    // Copy result from device memory to host memory
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    printf("result: %f,%f,%f,%f\n", h_A[0], h_A[1], h_A[N * N - 2], h_A[N * N - 1]);

    // Free device memory
    cudaFree(d_A);

    // Free host memory
    free(h_A);
}
