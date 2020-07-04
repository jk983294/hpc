#include <cuda_runtime.h>
#include <stdio.h>

constexpr size_t N = 512;

__global__ void add_one(size_t n, float* x) {
    int i = threadIdx.x;
    if (i < n) {
        x[i] = x[i] + 1;
    }
}

void switch_device() {
    // select device 0
    size_t size = N * sizeof(float);
    cudaSetDevice(0);  // Set device 0 as current
    float* p0;
    cudaMalloc(&p0, size);
    add_one<<<1, N>>>(N, p0);  // Launch kernel on device 0

    // switch to device 1
    cudaSetDevice(1);  // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);
    add_one<<<1, N>>>(N, p1);  // Launch kernel on device 1
}

void p2p_memory_access() {
    cudaSetDevice(0);  // Set device 0 as current
    float* p0;
    size_t size = N * sizeof(float);
    cudaMalloc(&p0, size);     // Allocate memory on device 0
    add_one<<<1, N>>>(N, p0);  // Launch kernel on device 0

    cudaSetDevice(1);                  // Set device 1 as current
    cudaDeviceEnablePeerAccess(0, 0);  // Enable peer-to-peer access with device 0

    // Launch kernel on device 1
    // This kernel launch can access memory on device 0 at address p0
    add_one<<<1, N>>>(N, p0);
}

void p2p_memory_copy() {
    cudaSetDevice(0);  // Set device 0 as current
    float* p0;
    size_t size = N * sizeof(float);
    cudaMalloc(&p0, size);  // Allocate memory on device 0

    cudaSetDevice(1);  // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);  // Allocate memory on device 1

    cudaSetDevice(0);          // Set device 0 as current
    add_one<<<1, N>>>(N, p0);  // Launch kernel on device 0

    cudaSetDevice(1);  // Set device 1 as current
    /**
     * implicit barrier
     * cudaMemcpyPeer starts after all commands on both devices finish
     * all commands on both devices issue after cudaMemcpyPeer finish
     */
    cudaMemcpyPeer(p1, 1, p0, 0, size);  // Copy p0 to p1
    add_one<<<1, N>>>(N, p1);            // Launch kernel on device 1
}

int main(void) {
    // enumerate these devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }

    switch_device();
    p2p_memory_access();
}
