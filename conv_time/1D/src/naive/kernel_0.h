// EXAMPLE

#ifndef KERNEL_H0_
#define KERNEL_H0_

__global__ void ConvKernel(int* I, int* F, int* R, int N, int M);

void Conv(int* I, int* F, int* R, int N, int M);

#endif // KERNEL_H0_
