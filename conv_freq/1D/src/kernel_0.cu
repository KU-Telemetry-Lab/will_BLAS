#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "kernel_1.h"

#define PI 3.14159265358979323846

using namespace std;

__global__ void PointWiseMultKernel(cufftComplex* I, cufftComplex* F, cufftComplex* R, int N) {
    // determine thread id
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // check boundary of the resulting vector
    if (thread_id >= N) return;

    float a = I[thread_id].x; 
    float b = I[thread_id].y;
    float c = F[thread_id].x; 
    float d = F[thread_id].y;

    R[thread_id].x = ((a * c) - (b * d)) / N; // scale to 1/N
    R[thread_id].y = ((a * d) + (b * c)) / N; // scale to 1/N
}


void PointWiseMult(cufftComplex* I, cufftComplex* F, cufftComplex* R, int N) {
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
    if ((N) > 512){
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }

    PointWiseMultKernel<<<blocksPerGrid, threadsPerBlock>>>(I, F, R, N);
}


void fft(cufftComplex* I, int N) {
    if (N <= 1) return;

    cufftComplex even[N/2];
    cufftComplex odd[N/2];

    for (int i = 0; i < N / 2; i++) {
        even[i].x = I[2 * i].x;
        even[i].y = I[2 * i].y;
        odd[i].x = I[2 * i + 1].x;
        odd[i].y = I[2 * i + 1].y;
    }

    fft(even, N/2);
    fft(odd, N/2);

    for (int k = 0; k < N / 2; k++) {
        float cos_theta = cos(-2 * PI * k / N);
        float sin_theta = sin(-2 * PI * k / N);

        float t_real = cos_theta * odd[k].x - sin_theta * odd[k].y;
        float t_imag = sin_theta * odd[k].x + cos_theta * odd[k].y;

        I[k].x = even[k].x + t_real;
        I[k].y = even[k].y + t_imag;

        I[k + N/2].x = even[k].x - t_real;
        I[k + N/2].y = even[k].y - t_imag;
    }
}


void ifft(cufftComplex* I, int N) {
    if (N <= 1) return;

    cufftComplex even[N/2];
    cufftComplex odd[N/2];

    for (int i = 0; i < N / 2; i++) {
        even[i].x = I[2 * i].x;
        even[i].y = I[2 * i].y;
        odd[i].x = I[2 * i + 1].x;
        odd[i].y = I[2 * i + 1].y;
    }

    // Recursive calls for even and odd components
    ifft(even, N/2);
    ifft(odd, N/2);

    for (int k = 0; k < N / 2; k++) {
        // Use positive sign for the IFFT exponent
        float cos_theta = cos(2 * PI * k / N);
        float sin_theta = sin(2 * PI * k / N);

        // Combine even and odd components
        float t_real = cos_theta * odd[k].x - sin_theta * odd[k].y;
        float t_imag = sin_theta * odd[k].x + cos_theta * odd[k].y;

        I[k].x = even[k].x + t_real;
        I[k].y = even[k].y + t_imag;

        I[k + N/2].x = even[k].x - t_real;
        I[k + N/2].y = even[k].y - t_imag;
    }
}


int is_power_two(int x) {
    return x && !(x & (x - 1));
}


int next_power_two(int x) {
    if (is_power_two(x)) return x;

    int power = 1;
    while (power < x) {
        power <<= 1;
    }
    return power;
}

void print_complex_vector(cufftComplex* I, int N) {
    printf("Real Vector = [");
    for (int i = 0; i < N; i++){
        printf("%f ", I[i].x);
    }
    printf("]\n\n");

    printf("Imaginary Vector = [");
    for (int i = 0; i < N; i++){
        printf("%f ", I[i].y);
    }
    printf("]\n\n");
}
