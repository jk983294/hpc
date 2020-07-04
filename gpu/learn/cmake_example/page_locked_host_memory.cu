#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>

/**
 * benefits:
 * copy between page-locked memory and device memory can be concurrent with kernel execution
 * it can be mapped into the address space of the device, eliminating the need to copy it to or from device memory
 * bandwidth is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as
 * write-combining
 *
 * shortcut: cudaMallocHost(&hostPtr, size); // allocates page-locked memory on the host
 */

// Add two vectors on the GPU
__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

int main(int argc, char **argv) {
    int idev = 0;               // use default device 0
    float *a, *b, *c;           // Pinned memory allocated on the CPU
    float *a_UA, *b_UA, *c_UA;  // Non-4K Aligned Pinned memory on the CPU
    float *d_a, *d_b, *d_c;     // Device pointers for mapped memory
    cudaDeviceProp deviceProp;

    printf("> Using Generic System Paged Memory (malloc)\n");

    checkCudaErrors(cudaSetDevice(idev));

    // Verify device supports mapped memory and set the device flags for mapping host memory
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));

    if (!deviceProp.canMapHostMemory) {
        fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", idev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

    /* Allocate mapped CPU memory. */

    int nelem = 1048576;
    size_t bytes = nelem * sizeof(float);

    a_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
    b_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
    c_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);

    // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
    a = (float *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
    b = (float *)ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
    c = (float *)ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

    checkCudaErrors(cudaHostRegister(a, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostRegister(b, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
    checkCudaErrors(cudaHostRegister(c, bytes, CU_MEMHOSTALLOC_DEVICEMAP));

    // Initialize the vectors
    for (int n = 0; n < nelem; n++) {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    // Get the device pointers for the pinned CPU memory mapped into the GPU memory space.
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));

    /* Call the GPU kernel using the CPU pointers residing in CPU mapped memory. */
    printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
    checkCudaErrors(cudaDeviceSynchronize());  // as kernel is async for mapped memory, we need sync here
    getLastCudaError("vectorAddGPU() execution failed");

    // Compare the results
    printf("> Checking the results from vectorAddGPU() ...\n");
    float errorNorm = 0.f, refNorm = 0.f;

    for (int n = 0; n < nelem; n++) {
        float ref = a[n] + b[n];
        float diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }
    errorNorm = (float)sqrt((double)errorNorm);
    refNorm = (float)sqrt((double)refNorm);

    if (errorNorm / refNorm < 1.e-6f) {
        printf("no error found\n");
    } else {
        printf("error found\n");
    }

    printf("> Releasing CPU memory...\n");

    checkCudaErrors(cudaHostUnregister(a));
    checkCudaErrors(cudaHostUnregister(b));
    checkCudaErrors(cudaHostUnregister(c));
    free(a_UA);
    free(b_UA);
    free(c_UA);

    cudaDeviceReset();
}
