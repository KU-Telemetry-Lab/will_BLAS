#ifndef KERNEL_CUH_
#define KERNEL_CUH_

__global__ void ConvKernel0(int* I, int* F, int* R, int N, int M);

void Conv0(int* I, int* F, int* R, int N, int M);

#endif
