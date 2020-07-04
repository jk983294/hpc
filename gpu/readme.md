## three key abstractions 
* a hierarchy of thread groups
* shared memories
* barrier synchronization

## kernel
square<<<#blocks, #thread per block, #Ns, stream>>>

\#thread per block 现在的架构最大1024个线程

\#Ns 每个block分配的 shared memory (in byte)

Thread blocks一定需要可以以任意顺序执行，Threads within a block可以通过Shared Memory以及barrier来合作

## block index
多维block: dim3(w, 1, 1) == dim3(w) == w

Each block within the grid can be identified through blockIdx variable

Each thread within the block can be identified through threadIdx variable

```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```

最大1024个线程 per block，所以上面 N*N，当N=32时就达到最大，如果矩阵大于1024个元素，那就需要多个block并行算

## Memory Hierarchy
Each thread has private local memory.

Each block has shared memory visible to all threads of the block

All threads have access to the same global memory

two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.

## Compute Capability

GeForce GTX 1050 	6.1

6 for devices based on the Pascal architecture

## CUDA runtime
The driver API provides
 * CUDA contexts - the analogue of host processes for the device 
 * CUDA modules - the analogue of dynamically loaded libraries for the device

During initialization, the runtime creates a CUDA context for each device in the system 

the CUDA context is shared among all the host threads of the application

### Asynchronous Concurrent Execution
following operations as independent tasks that can operate concurrently:

* Computation on the host;
* Computation on the device;
* Memory transfers from the host to the device;
* Memory transfers from the device to the host;
* Memory transfers within the memory of a given device;
* Memory transfers among devices.

#### stream
A stream is a sequence of commands that execute in order. 

Different streams may execute their commands out of order.

Kernel launches and host/device memory copies are issued to the default stream. They are therefore executed in order. 

##### Explicit Synchronization
* cudaDeviceSynchronize()
* cudaStreamSynchronize(stream)
* cudaStreamWaitEvent(stream, event)
* cudaStreamQuery()

##### Implicit Synchronization
Two commands from different streams cannot be concurrent if any below operations is issued in-between them:
* a page-locked host memory allocation,
* a device memory allocation,
* a device memory set,
* a memory copy between two addresses to the same device memory,
* any CUDA command to the NULL stream,
* a switch between the L1/shared memory configurations described in Compute Capability 3.x and Compute Capability 7.x.

#### graph
graph to be defined once and then launched repeatedly.

graph定义和执行分离可以 enables a number of optimizations

#### Error Checking
check for asynchronous errors: 
* checking the error code returned by cudaDeviceSynchronize()

cudaGetLastError() returns this variable and resets it to cudaSuccess

## multi-device
A kernel launch will fail if it is issued to a stream that is not associated to the current device

Each device has its own default stream 

On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections

#### Unified Virtual Address Space
All host memory allocations made via CUDA API calls and all device memory allocations on supported devices are within this virtual address range


#### Interprocess Communication
Any device memory pointer or event handle created by a host thread can be directly referenced by other threads within the same process

it cannot be directly referenced by threads belonging to a different process

To share device memory pointers and events across processes, an application must use the IPC API



## Compile
#### workflow
basic workflow consists in separating device code from host code and then:
* compiling the device code into an assembly form (PTX code),
* and modifying the host code by replacing the <<<...>>> syntax introduced in Kernels by the necessary CUDA runtime function calls to load and launch each compiled kernel from the PTX code.

##### Binary Compatibility
```sh
nvcc a.cu -code=sm_61
```

##### PTX Compatibility
```sh
nvcc a.cu arch=compute_61
```

Application Compatibility
##### PTX Compatibility
```sh
nvcc a.cu -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50
```

-arch=sm_35 is a shorthand for -arch=compute_35 -code=compute_35,sm_35

## Hardware Implementation
GPU = m个 Streaming Multiprocessors (SM) = m * n 个block = m * n * k 个thread

* 1 SM = n blocks
* 1 block = k threads
* 1 warp = 32 parallel threads

the blocks of the grid are enumerated and distributed to multiprocessors

The threads of a thread block execute concurrently on one multiprocessor

multiple thread blocks can execute concurrently on one multiprocessor

### SIMT
When a multiprocessor is given blocks to execute, it partitions them into warps and each warp gets scheduled by a warp scheduler for execution

Branch divergence occurs only within a warp

##### Independent Thread Scheduling
With Independent Thread Scheduling, the GPU maintains execution state per thread, 

including a program counter and call stack, and can yield execution at a per-thread granularity, 

either to make better use of execution resources or to allow one thread to wait for data to be produced by another

## Perf
memcpy

| :---: | :---: |
| method | latency |
| manual | 2.2ms |
| zero copy (managed/pinned memory) | 70.1ms |
| unified memory (UVA) | 85ms |

## sample
/usr/local/cuda-#.#/samples

## lib
Linux

libcudart.so, libcudart_static.a, libcudadevrt.a
libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a
libcublas.so, libcublas_static.a, libcublas_device.a
libnvblas.so
libcusparse.so, libcusparse_static.a
libcusolver.so, libcusolver_static.a
libcurand.so, libcurand_static.a
libnvgraph.so, libnvgraph_static.a
libnvjpeg.so, libnvjpeg_static.a

NVIDIA Performance Primitives Library

libnppc.so, libnppc_static.a, libnppial.so,
libnppial_static.a, libnppicc.so, libnppicc_static.a,
libnppicom.so, libnppicom_static.a, libnppidei.so,
libnppidei_static.a, libnppif.so, libnppif_static.a
libnppig.so, libnppig_static.a, libnppim.so,
libnppim_static.a, libnppist.so, libnppist_static.a,
libnppisu.so, libnppisu_static.a, libnppitc.so
libnppitc_static.a, libnpps.so, libnpps_static.a

libnvrtc.so, libnvrtc-builtins.so