#include <stdio.h>

__global__ void histogram(int n, int* color, int* bucket) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        int c = color[i];
        atomicAdd(&bucket[c], 1);
    }
}

void host_histogram() {
    int N = 1 << 20;
    int M = 1 << 10;
    int *color_, *bucket_, *d_color, *d_bucket;
    color_ = (int*)malloc(N * sizeof(int));
    bucket_ = (int*)malloc(M * sizeof(int));

    cudaMalloc(&d_color, N * sizeof(int));
    cudaMalloc(&d_bucket, M * sizeof(int));

    memset(bucket_, 0, M * sizeof(int));
    for (int i = 0; i < N; i++) {
        color_[i] = rand() % M;
        bucket_[color_[i]]++;
    }
    printf("cpu bucket: %d,%d,%d,%d,%d\n", bucket_[0], bucket_[1], bucket_[2], bucket_[3], bucket_[4]);
    memset(bucket_, 0, M * sizeof(int));

    cudaMemcpy(d_color, color_, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bucket, 0, M * sizeof(int));

    histogram<<<(N + 255) / 256, 256>>>(N, d_color, d_bucket);

    cudaMemcpy(bucket_, d_bucket, M * sizeof(int), cudaMemcpyDeviceToHost);

    printf("gpu bucket: %d,%d,%d,%d,%d\n", bucket_[0], bucket_[1], bucket_[2], bucket_[3], bucket_[4]);

    cudaFree(d_color);
    cudaFree(d_bucket);
    free(color_);
    free(bucket_);
}

/**
 * Introduce local maximums and update global only when new local maximum found
 */
__global__ void global_max(int* values, int* global_max, int* local_max, int num_locals) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int val = values[i];
    int li = i % num_locals;
    int old_max = atomicMax(&local_max[li], val);
    if (old_max < val) {
        atomicMax(global_max, val);
    }
}

void host_global_max() {
    int N = 1 << 20;
    int num_locals_ = 1 << 10;
    int *values_, *d_values, *d_local_max, *d_global_max;
    values_ = (int*)malloc(N * sizeof(int));

    cudaMalloc(&d_values, N * sizeof(int));
    cudaMalloc(&d_local_max, num_locals_ * sizeof(int));
    cudaMalloc(&d_global_max, sizeof(int));

    int h_global_max = -1;
    for (int i = 0; i < N; i++) {
        values_[i] = rand();
        if (h_global_max < values_[i]) h_global_max = values_[i];
    }
    printf("cpu global_max: %d\n", h_global_max);
    h_global_max = -1;

    cudaMemcpy(d_values, values_, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_local_max, 0, num_locals_ * sizeof(int));
    cudaMemset(d_global_max, 0, sizeof(int));

    global_max<<<(N + 255) / 256, 256>>>(d_values, d_global_max, d_local_max, num_locals_);

    cudaMemcpy(&h_global_max, d_global_max, sizeof(int), cudaMemcpyDeviceToHost);

    printf("gpu global_max: %d\n", h_global_max);

    cudaFree(d_values);
    cudaFree(d_local_max);
    cudaFree(d_global_max);
    free(values_);
}

int main(void) {
    host_histogram();
    host_global_max();
}
