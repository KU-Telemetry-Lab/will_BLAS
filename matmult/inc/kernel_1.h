#ifndef KERNEL_CUH_
#define KERNEL_CUH_

__global__ void MatMultKernel_1(float* A, float* B, float* C, int N);

void MatMult_1(float *A, float *B, float *C, int N);

#endif