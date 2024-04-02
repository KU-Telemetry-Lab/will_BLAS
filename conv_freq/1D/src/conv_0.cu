#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <chrono>
#include <cufft.h>
#include <cuda_runtime.h>

#include "kernel_1.h"


typedef struct {
    int length;
    cufftComplex* elements;
} VectorComplex;


int main(int argc, char **argv) {

    if (argc < 2) {
        printf("usage:  %s <y_length> <x_length>\n", argv[0]);
        exit(-1);
    }


    using clock = std::chrono::steady_clock;
    clock::time_point startTime;
    clock::time_point endTime;
    clock::duration allTime;

    uint64_t kernel_timeMsec;
    uint64_t cpu_timeMsec;


    int N = atoi(argv[1]); // length of input vector
    int M = atoi(argv[2]); // length of filter vector
    int R = N+M-1; // resulting vector size
    R = next_power_two(R); // bumping R to the next power of two


    // initialize host vectors h_I (input), h_F (filter), and h_R (result)
    VectorComplex h_I, h_F, h_R;
    h_I.length = h_F.length = h_R.length = R;
    size_t size_h = R * sizeof(cufftComplex);


    // allocate memory on host for h_I, h_F, and h_R
    h_I.elements = (cufftComplex*)malloc(size_h);
    h_F.elements = (cufftComplex*)malloc(size_h);
    h_R.elements = (cufftComplex*)malloc(size_h);


    // fill real vectors of h_I and h_F on host
    if (N >= M) {
        for (int i=0; i<N; i++){
            // h_I.elements[i] = sin(i);
            h_I.elements[i].x = i;
            if (i < M) {
                // h_F.elements[i] = rand() % 10;
                h_F.elements[i].x = 1;
            }
        }
        // zero pad remaining values in h_I and h_F
        for (int i = N; i < R; i++) {
            h_I.elements[i].x = 0;
            h_F.elements[i].x = 0;
        }
    } else {
        printf("Input length N (%d) greater than filter length M (%d)...\n", N, N);
        return 0;
    }

    // filling imaginary vectors of h_I, h_F, and h_R with zeros
    for (int i = 0; i < R; i++) {
        h_I.elements[i].y = 0;
        h_F.elements[i].y = 0;
        h_R.elements[i].y = 0;
    }
    // printf("INPUT VECTORS\n");
    // printf("h_I (input)\n");
    // print_complex_vector(h_I.elements, R);
    // printf("h_F (filter)\n");
    // print_complex_vector(h_F.elements, R);
    // printf("\n\n");

    // allocate device memory for d_I, d_F, and d_R
    VectorComplex d_I, d_F, d_R;
    d_I.length = d_F.length = d_R.length = R;
    size_t size_d = R * sizeof(cufftComplex);

// ####################################### FFT ON DEVICE #######################################
    startTime = clock::now();

    cudaMalloc(&d_I.elements, size_d);
    cudaMalloc(&d_F.elements, size_d);
    cudaMalloc(&d_R.elements, size_d);

    // Copy data from host to device
    cudaMemcpy(d_I.elements, h_I.elements, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F.elements, h_F.elements, size_d, cudaMemcpyHostToDevice);

    // Plan for CUBLAS fft implementation
    cufftHandle plan;
    cufftPlan1d(&plan, R, CUFFT_C2C, 1);

    // Execute FFT on h_I and h_F
    cufftExecC2C(plan, d_I.elements, d_I.elements, CUFFT_FORWARD);
    cufftExecC2C(plan, d_F.elements, d_F.elements, CUFFT_FORWARD);

// ################################# POINT WISE MULT ON DEVICE #################################

    // updated fft values already stored on device in d_I, d_F, and d_R
    PointWiseMult(d_I.elements, d_F.elements, d_R.elements, R);
    cudaDeviceSynchronize();
// ####################################### IFFT ON DEVICE ######################################

    // updated mult result values already stored on device in d_R
    cufftExecC2C(plan, d_R.elements, d_R.elements, CUFFT_INVERSE);

// ##################################### COPY BACK TO HOST #####################################

    // copy results back to host
    cudaMemcpy(h_R.elements, d_R.elements, size_d, cudaMemcpyDeviceToHost);

    endTime = clock::now();
    allTime = endTime - startTime;

    // printf("DEVICE OUTPUT VECTOR\n");
    // printf("h_R (device result)\n");
    // print_complex_vector(h_R.elements, R);

    // device conv timing results
    kernel_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms to convolve two a length %d and length %d complex array on DEVICE.\n", kernel_timeMsec, N, M);
    printf("\n\n");

// ########################### ERROR CHECKING AND CPU COMPARISON ###############################

    VectorComplex cpu_R;
    cpu_R.length = R;
    cpu_R.elements = (cufftComplex*)malloc(size_h);

    startTime = clock::now();

    fft(h_I.elements, R); // perform fft on h_I
    fft(h_F.elements, R); // perform fft on h_F

    // complex multiplication of fft(h_I) and fft(h_F)
    for (int i = 0; i < R; i++) {
        float a = h_I.elements[i].x; 
        float b = h_I.elements[i].y;
        float c = h_F.elements[i].x; 
        float d = h_F.elements[i].y;

        cpu_R.elements[i].x = ((a * c) - (b * d)) / R; // scale by 1/R
        cpu_R.elements[i].y = ((a * d) + (b * c)) / R; // scale by 1/R
    }

    ifft(cpu_R.elements, R);

    endTime = clock::now();
    allTime = endTime - startTime;

    // printf("HOST OUTPUT VECTOR\n");
    // printf("cpu_R (host result)\n");
    // print_complex_vector(cpu_R.elements, R);
    // printf("\n\n");

    // host conv timing results
    cpu_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms to convolve two a length %d and length %d complex array on HOST.\n", cpu_timeMsec, N, M);
    printf("\n\n");


    // doesn't handle float outputs very well so need to round of find another way to validate

    // // error checking
    // int error_count = 0;
    // for (int i = 0; i < R; i++){
    //     if ((h_R.elements[i].x != cpu_R.elements[i].x) || (h_R.elements[i].y != cpu_R.elements[i].y)){
    //         error_count += 1;   
    //         printf("ERROR in element #%d -> GPU: (%f+j%f) | CPU: (%f+j%f)\n", i, h_R.elements[i].x, h_R.elements[i].y, cpu_R.elements[i].x, cpu_R.elements[i].y);
    //     }
    // }
    // if (error_count != 0) {
    //     printf("Total errors = %d", error_count);
    // }

// #############################################################################################

    // Free host memory
    free(h_I.elements);
    free(h_F.elements);
    free(h_R.elements);
    free(cpu_R.elements);

    // Free device memory
    cudaFree(d_I.elements);
    cudaFree(d_F.elements);
    cudaFree(d_R.elements);

    // Destroy FFT plan
    cufftDestroy(plan);

    return 0;
}

