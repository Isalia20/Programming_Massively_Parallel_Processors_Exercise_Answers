#include <stdio.h>
#include <stdlib.h>



void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    int size = n * sizeof(float);

    float* A_d, B_d, C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    //kernel here
    //...
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    // vecAdd(A, B, C, N);
    int n = 100;
    float* A_d = (float*)malloc(n * sizeof(float));
    A_d[0] = 100.0;
    printf("%p\n", &A_d);
    printf("%p\n", *A_d);
    printf("%p\n", (void**)&A_d);
    printf("%p\n", some_addr);
    cudaMalloc((void**)&A_d, n * sizeof(float));

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    cudaFree(A_d);
}
