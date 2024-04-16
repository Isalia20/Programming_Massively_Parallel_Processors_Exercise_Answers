#include <stdio.h>
#include <stdlib.h>
#define CHANNELS 3

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

__global__
void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y + blockIdx.y + threadIdx.y;

    if (col < width && row < height){
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset]; // Red
        unsigned char g = Pin[rgbOffset + 1]; // Green
        unsigned char b = Pin[rgbOffset + 2]; // Blue
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h){
        int pixVal = 0;
        int pixels = 0;
        // Get average of the surrounding BLUR SIZE x BLUR SIZE box
        for (int blurRow =-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for (int blurCol =-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
                    pixVal += in[curRow * w + curCol];
                    ++pixels; // keep track of number of pixels for the average
                }
            }
        }
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

__global__
void MatMulKernel(float* M, float* N, float* P, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((row < width) && (col < width)){
        float Pvalue = 0;
        for (int k = 0; k < width; ++k){
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = Pvalue;
    }
}

__global__
void MatMulKernelRow(float* M, float* N, float* P, int width){
    int col = blockDimx. * blockIdx.x + threadIdx.x;

    if ((col < width)){
        float Pvalue = 0;
        for (int i = 0; i < width; ++i){
            Pvalue += M[]
            P[row * width + col] = Pvalue;
        }
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    float* A_d, *B_d, *C_d;
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(128, 4, 2);
    int size = n * sizeof(float);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(C_d);
}

int main(){
    int n = 100;
    float* A_h = (float*)malloc(n * sizeof(float));
    float* B_h = (float*)malloc(n * sizeof(float));
    float* C_h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++){
        A_h[i] = 1.0;
        B_h[i] = 2.0;
        C_h[i] = 0.0;
    }
    vecAdd(A_h, B_h, C_h, 100);
    for (int i = 0; i < n; i++){
        printf("%f\n", C_h[i]);
    }
    return 0;
}