#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <time.h>

#include "kernel_0.h"
#define FILTER_LENGTH 512


typedef struct {
    int length;
    int* elements;
} Vector;


void validate_result(int *I, int *F, int *R, int N, int M) {
    // I = input vector of length N
    // F = filter vector of length M
    // R = GPU results vector of length N+M-1

    for (int i = 0; i < N + M - 1; i++) {
        int temp = 0;

        // determine the start and end indices for the convolution operation
        int start = i >= M ? i - M + 1 : 0;
        int end = i < N ? i : N - 1;

        for (int j = start; j <= end; j++) {
            if((i-j) >= 0 && (i-j) < M) {
                temp += I[j] * F[(i-j)];
            }
        }
        // ensure temp matches result from GPU
        // printf("%d", R[i]);
        assert(temp == R[i]);
    }
}


int main(int argc, char **argv) {

    using clock = std::chrono::steady_clock;
    clock::time_point startTime;
    clock::time_point endTime;
    clock::duration allTime;
    uint64_t timeMsec;

    if (argc < 2) {
        printf("usage:  %s <input length>\n", argv[0]);
        exit(-1);
    }


    int N = atoi(argv[1]); // length of input vector
    int M = FILTER_LENGTH; // length of filter vector
    int R = N+FILTER_LENGTH-1; // resulting vector size


    // initialize host vectors h_I (input), h_F (filter), and h_R (result)
    Vector h_I;
    h_I.length = N;
    size_t size_h_I = h_I.length * sizeof(int);

    Vector h_F;
    h_F.length = M;
    size_t size_h_F = h_F.length * sizeof(int);

    Vector h_R;
    h_R.length = R;
    size_t size_h_R = h_R.length * sizeof(int);


    // allocate memory on host for h_I, h_F, and h_R
    h_I.elements = (int*)malloc(size_h_I);
    h_F.elements = (int*)malloc(size_h_F);
    h_R.elements = (int*)malloc(size_h_R);


    // fill vectors h_I and h_F on host
    if (N >= M) {
        for (int i=0; i<N; i++){
            // h_I.elements[i] = sin(i);
            h_I.elements[i] = 1;
            if (i <= M) {
                // h_F.elements[i] = rand() % 10;
                h_F.elements[i] = 1;
            }
        }
    } else {
        printf("Input length N (%d) greater than filter length M (%d)...\n", N, N);
        return 0;
    }

    // initialize device vectors d_I (input), d_F (filter), and d_R (result)
    Vector d_I;
    d_I.length = N;
    size_t size_d_I = d_I.length * sizeof(int);

    Vector d_F;
    d_F.length = M;
    size_t size_d_F = d_F.length * sizeof(int);
    
    Vector d_R;
    d_R.length = R;
    size_t size_d_R = d_R.length * sizeof(int);

    // allocate memory on device  for d_I, d_F, and d_R
    cudaMalloc(&d_I.elements, size_d_I);
    cudaMalloc(&d_F.elements, size_d_F);
    cudaMalloc(&d_R.elements, size_d_R);

    startTime = clock::now();

    // copy h_I and h_F from host memory to d_I and d_F in device memory
    cudaMemcpy(d_I.elements, h_I.elements, size_d_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F.elements, h_F.elements, size_d_F, cudaMemcpyHostToDevice);

    // call 1D convolution kernel helper function
    Conv(d_I.elements, d_F.elements, d_R.elements, N, M);
    // wait for all thread blocks to execute
    cudaDeviceSynchronize();

    // copy d_R from device memory to h_R in host memory
    cudaMemcpy(h_R.elements, d_R.elements, size_h_R, cudaMemcpyDeviceToHost);

    endTime = clock::now();
    allTime = endTime - startTime;
    timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu microsec for 1D convolution on GPU.\n", timeMsec);

    // validate result on host
    validate_result(h_I.elements, h_F.elements, h_R.elements, N, M);

    // free device memory
    cudaFree(d_I.elements);
    cudaFree(d_F.elements);
    cudaFree(d_R.elements);

    // free host memory
    free(h_I.elements);
    free(h_F.elements);
    free(h_R.elements);
}