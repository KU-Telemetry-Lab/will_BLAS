#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#include "kernel_1.h"
#define FILTER_LENGTH 512


__constant__ int F[FILTER_LENGTH];

__global__ void ConvKernel(int* I, int* R, int filter_length, int padded_length) {
    // determine thread id
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // check boundary of the resulting vector
    if (thread_id >= padded_length) return;

    int temp = 0;        
    // step through each element in the filter
    for (int i = 0; i < filter_length; i++) {
        // calculate the corresponding index in the input vector
        int inputIdx = thread_id - i;
        // check if the index is within the bounds of the input vector
        if (inputIdx >= 0 && inputIdx < padded_length) {
            temp += I[inputIdx] * F[i];
        }
    }
    // write back results
    R[thread_id] = temp;
}


void Conv(int* I, int *R, int filter_length, int padded_length) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (padded_length + threadsPerBlock -1) / threadsPerBlock;

    ConvKernel<<<blocksPerGrid, threadsPerBlock>>>(I, R, filter_length, padded_length);
}


void FillConstant(int* I) {
    cudaMemcpyToSymbol(F, I, FILTER_LENGTH * sizeof(int));
}