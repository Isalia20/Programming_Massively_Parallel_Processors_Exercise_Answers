// 1. In this chapter we implemented a matrix multiplication kernel that has each
// thread produce one output matrix element. In this question, you will
// implement different matrix-matrix multiplication kernels and compare them.
// a. Write a kernel that has each thread produce one output matrix row. Fill in
// the execution configuration parameters for the design.
// b. Write a kernel that has each thread produce one output matrix column. Fill
// in the execution configuration parameters for the design.
// c. Analyze the pros and cons of each of the two kernel designs

// For square matrices
__global__ 
void MatMulKernelRow(float* M, float* N, float* P, int width){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < width){
        for (int col = 0; col < width; ++col){
            float Pvalue = 0.0;
            for (int k = 0; k < width; ++k){
                Pvalue += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = Pvalue;
        }
    }
}

__global__
void MatMulKernelCol(float* M, float* N, float* P, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width){
        for (int row = 0; row < width; ++row){
            float Pvalue = 0.0;
            for (int k = 0; k < width; ++k){
                Pvalue += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = Pvalue;
        }
    }
}
// For non-square matrices
__global__
void MatmulKernelRow(float* M, float* N, float* P, int height_M, int width_M, int width_N){
    // Note that it doesn't require height_N since its assumed that height_N is the same as width_M(otherwise matmul won't work)
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height_M){
        for (int col = 0; col < width_N; ++col){
            float Pvalue = 0.0;
            for (int k = 0; k < width_N; ++k){
                Pvalue += M[row * width_M + k] * N[k * width_N + col];
            }
            P[row * width_N + col] = Pvalue;
        }
    }
}

// Extra matmul in C(without CUDA)(just for practicing)
void MatMul(float* M, float* N, float* P, int height_M, int width_M, int width_N){
    for (int row = 0; row < height_M; ++row){
        for (int col = 0; col < width_N; ++col){
            float inner_product = 0.0;
            for (int k = 0; k < width_M; ++k){
                inner_product += M[row * width_M + k] * N[width_N * k + col];
            }
            P[row * width_N + col] = inner_product;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// 2.A matrix-vector multiplication takes an input matrix B and a vector C and
// produces one output vector A. Each element of the output vector A is the dot
// product of one row of the input matrix B and C, that is, A[i] 5 Pj B[i][j] 1 C[j].
// For simplicity we will handle only square matrices whose elements are single-
// precision floating-point numbers. Write a matrix-vector multiplication kernel and
// the host stub function that can be called with four parameters: pointer to the output
// matrix, pointer to the input matrix, pointer to the input vector, and the number of
// elements in each dimension. Use one thread to calculate an output vector element.

// Let's say M is [A, B] matrix and N is [B, 1], P will be [A, 1]
//---------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

__global__
void MatVecMulKernel(float* M, float* N, float* P, int width, int height) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height) {
        float sum = 0.0;
        for (int col = 0; col < width; col++) {
            sum += M[row * width + col] * N[col];
        }
        P[row] = sum;
    }
}

void MatVecMul(float* M_h, float* N_h, float* P_h, int width, int height) {
    float *M_d, *N_d, *P_d;

    cudaMalloc((void**)&M_d, width * height * sizeof(float));
    cudaMalloc((void**)&N_d, width * sizeof(float));
    cudaMalloc((void**)&P_d, height * sizeof(float));

    cudaMemcpy(M_d, M_h, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, width * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(ceil(height / 128.0), 1, 1);
    dim3 dimBlock(128, 1, 1);
    MatVecMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width, height);

    cudaMemcpy(P_h, P_d, height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int width = 100;
    int height = 200;
    float* M_h = (float*)malloc(width * height * sizeof(float));
    float* N_h = (float*)malloc(width * sizeof(float));
    float* P_h = (float*)malloc(height * sizeof(float));


    // [A, B]  [B, 1] [A, 1]
    for (int i = 0; i < width * height; i++) {
        M_h[i] = 1.0; // Example values
    }
    for (int i = 0; i < width; i++) {
        N_h[i] = 2.0; // Example values
    }
    
    MatVecMul(M_h, N_h, P_h, width, height);

    for (int i = 0; i < height; i++) {
        printf("%f\n", P_h[i]);
    }

    free(M_h);
    free(N_h);
    free(P_h);
    return 0;
}
//---------------------------------------------------------------------------------
// 3. 
// What is the number of threads per block? 
// Number of threads per block is 16 * 32
// What is the number of threads in the grid?
// 16 * 32 * num_blocks((300 - 1) / 16 + 1 * (150 - 1) / 32 + 1);
// 16 * 32 * 19 * 5 = 48640
// What is the number of blocks in the grid?
// ((300 - 1) / 16 + 1 * (150 - 1) / 32 + 1) = 19 * 5 = 95
// What is the number of threads that execute the code on line 05?
// Since some of the threads get filtered out we can calculate number of threads that come on that line
// by N * M which is 45000, so extra 3640 threads were created
//---------------------------------------------------------------------------------
// 4.
// W = 400 H = 500, 1D array
// Array index element at row 20 and column 10
// Row major order
// Given that row 0 exists, then formula will be:
// 20 * W + 10
// Column major order
// Given that column 0 exists then formula will be:
// 10 * H + 20
//---------------------------------------------------------------------------------
// 5. W = 400 H = 500 D = 300
// row major order
// array index of the tensor of the element at x = 10 y = 20 z = 5
// given x is column, y is row and z is depth
// we want element on row 10, column 20 and depth 5
// formula will be:
// H * W * z + y * W + x

