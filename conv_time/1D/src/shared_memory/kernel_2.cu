#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#include "kernel_2.h"
#define FILTER_LENGTH 512


__constant__ int F[FILTER_LENGTH];

__global__ void ConvKernel(int* I, int* R, int filter_length, int padded_length) {
    // determine thread id
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // store elements needed to compute output in shared memory
    extern __shared__ int shared[];

     // load elements from input vector to shared, checking boundary
    if (thread_id < padded_length) {
        shared[threadIdx.x] = I[thread_id];
    } else {
        // padding with 0 for threads beyond padded_length
        shared[threadIdx.x] = 0;
    }
    __syncthreads(); // ensure all loads are complete

    // only proceed if thread_id within boundary of result
    if (thread_id >= padded_length) return;

    int temp = 0;        
    // step through each element in the filter
    for (int i = 0; i < filter_length; i++) {
        int shared_id = threadIdx.x - i;
        if (shared_id >= 0 && shared_id < blockDim.x) {
            // value is in shared memory
            temp += shared[shared_id] * F[i];
        } else {
            // value is outside shared memory (global memory access)
            temp += I[thread_id - i] * F[i];
        }
    }
    // write back results
    R[thread_id] = temp;
}


void Conv(int* I, int *R, int filter_length, int padded_length) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (padded_length + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(int);

    ConvKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(I, R, filter_length, padded_length);
}

void FillConstant(int* I) {
    cudaMemcpyToSymbol(F, I, FILTER_LENGTH * sizeof(int));
}