#define TILE_WIDTH 16
__global__
void matrixMulKernel(float* M, float* N, float* P, int width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0;
    for (int i = 0; i < width / TILE_WIDTH; i++){
        Mds[ty][tx] = M[i * TILE_WIDTH + Row * width + tx];
        Nds[ty][tx] = N[i * TILE_WIDTH * width + ty * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

int main(){}