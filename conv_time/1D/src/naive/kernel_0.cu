#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "kernel_0.h"

__global__ void ConvKernel(int* I, int* F, int* R, int N, int M) {
    // I = input vector of length N
    // F = filter vector of length M
    // R = results vector of length N+M-1

    // determine thread id
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // check boundary of the resulting vector
    if (thread_id >= N+M-1) return;

    int temp = 0;        
    // step through each element in the filter
    for (int i = 0; i < M; i++) {
        // calculate the corresponding index in the input vector
        int inputIdx = thread_id - i;
        // check if the index is within the bounds of the input vector
        if (inputIdx >= 0 && inputIdx < N) {
            temp += I[inputIdx] * F[i];
        }
    }

    // write back results
    R[thread_id] = temp;
}


void Conv(int* I, int* F, int *R, int N, int M) {
    dim3 threadsPerBlock(N+M-1);
    dim3 blocksPerGrid(1);
    if ((N+M-1) > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N+M-1)/double(threadsPerBlock.x));
    }

    ConvKernel<<<blocksPerGrid, threadsPerBlock>>>(I, F, R, N, M);
}

