#include <cuda_runtime.h>
#include <stdio.h>

constexpr int numThreadsPerBlock = 1024;

__global__ void reduce0(int *input, int *output) {
    __shared__ int sdata[numThreadsPerBlock];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = input[i];

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];  // merge sum
        }

        __syncthreads();  // wait for other thread merge finish, then start next round merge
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main(void) {
    int *deviceInput;
    int *deviceOutput;

    int numInputElements = 64 * 64;
    int numOutputElements;  // number of elements in the output list, initialised below

    numOutputElements = numInputElements / (numThreadsPerBlock / 2);
    if (numInputElements % (numThreadsPerBlock / 2)) {
        numOutputElements++;
    }

    printf("input=%d, output=%d, numThreadsPerBlock=%d\n", numInputElements, numOutputElements, numThreadsPerBlock);

    int *hostInput = (int *)malloc(numInputElements * sizeof(int));
    int *hostOutput = (int *)malloc(numOutputElements * sizeof(int));

    for (int i = 0; i < numInputElements; ++i) {
        hostInput[i] = 1;
    }

    const dim3 blockSize(numThreadsPerBlock, 1, 1);
    const dim3 gridSize(numOutputElements, 1, 1);

    cudaMalloc(&deviceInput, numInputElements * sizeof(int));
    cudaMalloc(&deviceOutput, numOutputElements * sizeof(int));

    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(int), cudaMemcpyHostToDevice);

    reduce0<<<gridSize, blockSize>>>(deviceInput, deviceOutput);

    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(int), cudaMemcpyDeviceToHost);

    for (int ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];  // accumulates the sum in the first element
    }

    int sumGPU = hostOutput[0];

    printf("GPU Result: %d\n", sumGPU);
}
