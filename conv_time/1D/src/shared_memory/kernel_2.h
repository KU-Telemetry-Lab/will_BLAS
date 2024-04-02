

#ifndef KERNEL_H2_
#define KERNEL_H2_

__global__ void ConvKernel(int* I, int* R, int filter_length, int padded_length);

void Conv(int* I, int *R, int filter_length, int padded_length);

void FillConstant(int* I);

#endif // KERNEL_H2_