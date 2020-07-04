#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MyKernel(int *d, int *a, int *b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

__global__ void TestKernel(int *array, int arrayCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount) {
    int blockSize;    // The launch configurator returned block size
    int minGridSize;  // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;     // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)MyKernel, 0, arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    TestKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with cudaOccupancyMaxActiveBlocksPerMultiprocessor
    return 0;
}

// Host code
int main() {
    int numBlocks;  // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, 0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    printf("numBlocks=%d, blockSize=%d, warpSize=%d, maxThreadsPerProcessor=%d\n", numBlocks, blockSize, prop.warpSize,
           prop.maxThreadsPerMultiProcessor);
    printf("Occupancy: %f\n", (double)activeWarps / maxWarps);

    return 0;
}
