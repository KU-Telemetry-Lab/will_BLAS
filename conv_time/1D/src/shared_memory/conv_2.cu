#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <time.h>

#include "kernel_2.h"
#define FILTER_LENGTH 512

typedef struct {
    int* elements;
} Vector;


void validate_result(int *I, int *H, int *R, int N, int M) {
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
                temp += I[j] * H[(i-j)];
            }
        }
        // ensure temp matches result from GPU
        // printf("%d", R[i]);
        assert(temp == R[i]);
        // if (temp != R[i]) {
        //     printf("temp: %d | R: %d\n", temp, R[i]);
        // }
    }
}


int next_power_two(int x) {
    if ((x && !(x & (x - 1)))) return x;

    int power = 1;
    while (power < x) {
        power <<= 1;
    }
    return power;
}


void print_vector(int* I, int N) {
    printf(" Vector = [");
    for (int i = 0; i < N; i++){
        printf("%d ", I[i]);
    }
    printf("]\n\n");
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
    int padded_length = next_power_two(R); // bumping R to the next power of two


    // initialize host vectors h_I (input), h_F (filter), and h_R (result)
    Vector h_I, h_F, h_R;
    size_t size_h = padded_length * sizeof(int);

    // allocate memory on host for h_I, h_F, and h_R
    h_I.elements = (int*)malloc(size_h);
    h_F.elements = (int*)malloc(size_h);
    h_R.elements = (int*)malloc(size_h);


    // fill real vectors of h_I and h_F on host
    if (N >= M) {
        for (int i=0; i<N; i++){
            // h_I.elements[i] = sin(i);
            h_I.elements[i] = 1;
            if (i < M) {
                // h_F.elements[i] = rand() % 10;
                h_F.elements[i] = 1;
            }
        }
        // zero pad remaining values in h_I and h_F
        for (int i = N; i < padded_length; i++) {
            h_I.elements[i] = 0;
            h_F.elements[i] = 0;
        }
    } else {
        printf("Input length N (%d) greater than filter length M (%d)...\n", N, N);
        return 0;
    }

    // initialize device vectors d_I (input) and d_R (result)
    Vector d_I, d_R;
    size_t size_d = padded_length * sizeof(int);

    // allocate memory on device  for d_I and d_R
    cudaMalloc(&d_I.elements, size_d);
    cudaMalloc(&d_R.elements, size_d);

    startTime = clock::now();

    // copy h_I and h_F from host memory to d_I and d_F in device memory
    cudaMemcpy(d_I.elements, h_I.elements, size_d, cudaMemcpyHostToDevice);

    // copy the data directly to the symbol (no offset)
    FillConstant(h_F.elements);

    // call 1D convolution kernel helper function
    Conv(d_I.elements, d_R.elements, M, padded_length);
    // wait for all thread blocks to execute
    cudaDeviceSynchronize();

    // copy d_R from device memory to h_R in host memory
    cudaMemcpy(h_R.elements, d_R.elements, size_h, cudaMemcpyDeviceToHost);

    endTime = clock::now();
    allTime = endTime - startTime;
    timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu microsec for 1D convolution on GPU.\n", timeMsec);

    // validate result on host
    validate_result(h_I.elements, h_F.elements, h_R.elements, N, M);

    // free device memory
    cudaFree(d_I.elements);
    cudaFree(d_R.elements);

    // free host memory
    free(h_I.elements);
    free(h_F.elements);
    free(h_R.elements);
}