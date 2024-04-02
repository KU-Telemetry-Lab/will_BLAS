#ifndef KERNEL_CUH_
#define KERNEL_CUH_

// kernel functions
__global__ void PointWiseMultKernel(cufftComplex* I, cufftComplex* H, cufftComplex* R, int N);

// host functions
void PointWiseMult(cufftComplex* I, cufftComplex* H, cufftComplex* R, int N);

void fft(cufftComplex* I, int N);

void ifft(cufftComplex* I, int N);

int is_power_two(int x);

int next_power_two(int x);

void print_complex_vector(cufftComplex* I, int N);

#endif
