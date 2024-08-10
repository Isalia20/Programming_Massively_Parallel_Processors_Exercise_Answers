#include <stdio.h>
#define TILE_WIDTH 16

__global__ void matMulKernel(float* A_d, float* B_d, float* C_d, int num_rows, int num_cols){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int Row = blockIdx.y * TILE_WIDTH + ty;
    const int Col = blockIdx.x * TILE_WIDTH + tx;
    // initialize shared memory matrices
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Copy from global memory to tiled memory
    float Pval = 0.0;
    for (int i = 0; i < num_cols / TILE_WIDTH; i++){
        // get tiles into shared memory
        As[ty][tx] = A_d[i * TILE_WIDTH + Row * num_cols + tx];
        Bs[tx][ty] = B_d[i * TILE_WIDTH + Col * num_rows + ty]; // B_d is column stored
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++){
            Pval += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C_d[Row * num_cols + Col] = Pval;
}


void free_memory(float* A, float* B, float* C, float* A_d, float* B_d, float* C_d){
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);
}

int main(){
    // init matrices on cpu
    float *A, *B, *C;
    int num_rows = 16384;
    int num_cols = 16384;
    int numel = num_rows * num_cols;
    int alloc_size = numel * sizeof(float);
    A = (float*)malloc(alloc_size); // imagine row stored matrix
    B = (float*)malloc(alloc_size); // imagine column stored matrix
    C = (float*)malloc(alloc_size); // we allocate it with same size as its same matrices
    
    // Initialize A and B
    for (int i = 0; i < numel; i++) {A[i] = 1.0;};
    for (int i = 0; i < numel; i++) {B[i] = 1.0;};

    // init pointers on gpu
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, alloc_size);
    cudaMalloc((void**)&B_d, alloc_size);
    cudaMalloc((void**)&C_d, alloc_size);
    cudaMemcpy(A_d, A, alloc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, alloc_size, cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, alloc_size); // initialize matrix C to 0s

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((num_rows + blockSize.x - 1) / blockSize.x, (num_cols + blockSize.y - 1) / blockSize.y);
    matMulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, num_rows, num_cols);
    cudaMemcpy(C, C_d, alloc_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++){
        printf("%f \n", C[i]);
    }
    free_memory(A, B, C, A_d, B_d, C_d);
}