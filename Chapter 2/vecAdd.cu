#include <stdio.h>
#include <stdlib.h>


__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}


void vectAdd(float* A, float* B, float* C, int n){
    // A_d, B_d, C_d allocations and copies ommited
    //...
    float* A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Launch ceil(n/256) blocks of 256 threads each
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    float* A, *B, *C;
    A = (float*)malloc(100000 * sizeof(float));
    B = (float*)malloc(100000 * sizeof(float));
    C = (float*)malloc(100000 * sizeof(float));
    for (int i = 0; i < 100000; ++i){
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }
    vectAdd(A, B, C, 10);
    free(A);
    free(B);
    free(C);
    return 0;
}
