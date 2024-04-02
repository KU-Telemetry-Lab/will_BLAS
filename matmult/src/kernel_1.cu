#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "kernel_1.h"

#define BLOCKSIZE 32

__global__ void MatMultKernel_1(float* A, float* B, float* C, int N) {

    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.y % BLOCKSIZE);

    float tmpSum = 0.0;

    if (x < N && y < N) {
        for (int i = 0; i < N; i++) {
            tmpSum += A[x * N + i] * B[i * N + y];
        }
    }
    C[x * N + y] = tmpSum;
}

void MatMult_1(float *A, float *B, float *C, int N){
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blocksPerGrid((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);

    MatMultKernel_1<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}
