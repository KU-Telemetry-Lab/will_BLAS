#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#include "kernel_1.h"

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


int main(int argc, char **argv)
{
    // initializing timing parameters for recording
    // cpu and gpu speeds
    using clock = std::chrono::steady_clock;
    clock::time_point startTime;
    clock::time_point endTime;
    clock::duration allTime;
    uint64_t memcpy_timeMsec;
    uint64_t memrd_timeMsec;
    uint64_t kernel_timeMsec;
    uint64_t total_timeMsec;
    uint64_t cpu_timeMsec;


    if (argc < 2) {
        printf("usage:  %s <maxtrix dim>\n", argv[0]);
        exit(-1);
    }

    int N;
    N = atoi(argv[1]);


    // initialize host matricies h_A, h_B, and h_C on host
    Matrix h_A;
    h_A.width = N;
    h_A.height = N;
    size_t size_h_A = h_A.height * h_A.width * sizeof(float);

    Matrix h_B;
    h_B.width = N;
    h_B.height = N;
    size_t size_h_B = h_B.height * h_B.width * sizeof(float);

    Matrix h_C;
    h_C.width = N;
    h_C.height = N;
    size_t size_h_C = h_C.height * h_C.width * sizeof(float);


    // allocate memory on host for h_A, h_B, and h_C
    h_A.elements = (float*)malloc(size_h_A);
    h_B.elements = (float*)malloc(size_h_B);
    h_C.elements = (float*)malloc(size_h_C);


    // fill matricies h_A and h_B on host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            // h_A[i][j]
            h_A.elements[i*N+j] = sin(i);
            h_B.elements[i*N+j] = cos(j);
        }
    }


    // initialize matricies d_A, d_B, and d_C on device
    Matrix d_A;
    d_A.width = h_A.width;
    d_A.height = h_A.height;
    size_t size_d_A = h_A.width * h_A.height * sizeof(float);

    Matrix d_B;
    d_B.width = h_B.width; 
    d_B.height = h_B.height;
    size_t size_d_B = h_B.width * h_B.height * sizeof(float);

    Matrix d_C;
    d_C.width = h_C.width; 
    d_C.height = h_C.height;
    size_t size_d_C = h_C.width * h_C.height * sizeof(float);


    // allocate memory on decive for d_A, d_B, and d_C (result)
    cudaMalloc(&d_A.elements, size_d_A);
    cudaMalloc(&d_B.elements, size_d_B);
    cudaMalloc(&d_C.elements, size_d_C);


    // copy h_A and h_B from host memory to d_A and d_B in device memory (and time it)
    startTime = clock::now();
    cudaMemcpy(d_A.elements, h_A.elements, size_d_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, h_B.elements, size_d_B, cudaMemcpyHostToDevice);
    endTime = clock::now();
    allTime = endTime - startTime;

    // memcpy timing results
    memcpy_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms to copy 2 %dx%d matricies from CPU memory to GPU memory.\n", memcpy_timeMsec, N, N);



    // perform matrix multiplication on device (and time it)
    startTime = clock::now();
    MatMult_1(d_A.elements, d_B.elements, d_C.elements, N);
    // wait for all threads to complete
    cudaDeviceSynchronize();
    endTime = clock::now();
    allTime = endTime - startTime;

    // kernel timing results
    kernel_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms to perform %dx%d matrix multiply on GPU.\n", kernel_timeMsec, N, N);


    // copy d_C from device memory to h_C in host memory (and time it)
    startTime = clock::now();
    cudaMemcpy(h_C.elements, d_C.elements, size_h_C, cudaMemcpyDeviceToHost);
    endTime = clock::now();
    allTime = endTime - startTime;

    // memcpy timing results
    memrd_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms to copy 1 %dx%d matrix from GPU memory to CPU memory.\n\n", memrd_timeMsec, N, N);

    total_timeMsec = memcpy_timeMsec + kernel_timeMsec + memrd_timeMsec;
    printf("%8lu ms for full %dx%d matrix multiply and data transfers.\n\n", total_timeMsec, N, N);

    // ERROR CHECKING AND CPU COMPARISON #############################################
    // initialize matrix cpu_C to hold cpu results
    Matrix cpu_C;
    cpu_C.width = N;
    cpu_C.height = N;
    size_t size_cpu_C = cpu_C.height * cpu_C.width * sizeof(float);

    // allocate memory on host for cpu_C
    cpu_C.elements = (float*)malloc(size_cpu_C);

    // performing matrix multiplication on CPU (and time it)
    startTime = clock::now();
    float sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
        sum = 0.f;
        for (int n=0; n<N; n++){
            sum += h_A.elements[row*N+n] * h_B.elements[n*N+col];
        }
        cpu_C.elements[row*N+col] = sum;
        }
    }
    endTime = clock::now();
    allTime = endTime - startTime;
    cpu_timeMsec = std::chrono::duration_cast<std::chrono::microseconds>(allTime).count();
    printf("%8lu ms for full %dx%d matrix multiply on CPU.\n\n\n", cpu_timeMsec, N, N);
    
    // error checking
    int error_count = 0;
    for (int i = 0; i < (N*N); i++){
        //  need to fix found value...
        float epsilon = .0001;
        if (fabs(h_C.elements[i] - cpu_C.elements[i]) > epsilon) {
            error_count += 1;   
            printf("element %d GPU: %f | CPU: %f\n", i, h_C.elements[i], cpu_C.elements[i]);
        }
    }


    printf("Total errors in GPU matrix: %d\n", error_count);


    // free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    // free host memory
    free(h_A.elements);
    free(h_B.elements);
    free(h_C.elements);
    free(cpu_C.elements);
}