#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_one(int n, float* x, float* y) {
    int i = threadIdx.x;
    if (i < n) {
        y[i] = x[i] + 1;
    }
    // printf("thread %d finish\n", i);
}

void initialize_input(float* h_A, int n) {
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
    }
}

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* data) {
    printf("inside callback %zu\n", (size_t)data);
}

int main(void) {
    int N = 4 * 4;
    size_t size = N * sizeof(float);

    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream[i]);
    float* hostPtr[2];
    cudaMallocHost(&hostPtr[0], size);  // allocates page-locked memory on the host
    cudaMallocHost(&hostPtr[1], size);

    initialize_input(hostPtr[0], N);
    initialize_input(hostPtr[1], N);

    // Allocate vectors in device memory
    float* devPtrIn[2];
    float* devPtrOut[2];
    cudaMalloc(&devPtrIn[0], size);
    cudaMalloc(&devPtrIn[1], size);
    cudaMalloc(&devPtrOut[0], size);
    cudaMalloc(&devPtrOut[1], size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (size_t i = 0; i < 2; ++i) {
        cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
        add_one<<<1, N, 0, stream[i]>>>(N, devPtrIn[i], devPtrOut[i]);
        cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
        cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("elapsedTime=%f\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < 2; ++i) cudaStreamSynchronize(stream[i]);

    // Free device memory
    cudaFree(&devPtrIn[0]);
    cudaFree(&devPtrIn[1]);
    cudaFree(&devPtrOut[0]);
    cudaFree(&devPtrOut[1]);

    for (int i = 0; i < 2; ++i) cudaStreamDestroy(stream[i]);

    printf("result: %f,%f\n", hostPtr[0][3], hostPtr[1][3]);

    // Free host memory
    cudaFreeHost(hostPtr[0]);
    cudaFreeHost(hostPtr[1]);
}
