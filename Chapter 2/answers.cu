// 1. C
// Explanation: 
// i = blockIdx.x * blockDim.x + threadIdx.x
// is the basic way to retrieve an index for 1D setup of blocks and threads

// 2. C
// Explanation:
// i=(blockIdx.x * blockDim.x + threadIdx.x)*2;
// Imagine blockIdx.x = 0 blockDim.x = 256 and threadIdx.x = 0, then i=0;
// if threadIdx.x = 1 then i = 2; which is exactly what we want.
// Adjacent data can be retrieved with i and i + 1 in a single thread. Ex:
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = (threadIdx.x + blockDim.x * blockIdx.x) * 2;
    if (i < n){
        C[i] = A[i] + B[i];
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

// 3. D
// Explanation:
// i=blockIdx.x * blockDim.x * 2 + threadIdx.x;
// Imagine blockIdx.x = 0, blockDim.x=256 and threadIdx.x = 0; then i = 0;
// if blockIdx.x = 1 then i becomes 1 * 256 * 2 + 0 = 512;
// Meaning all the elements before 512 were processed by blockIdx.x = 0;
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i]; // Imagine i = 0, then indexing here is 0
    }

    int j = i + blockDim.x;
    if (j < n){
        C[j] = A[j] + B[j];
    }
}

// 4. C
// Explanation: 1024 threads, 8000 / 1024 = 7.81 and we want to create more threads than needed and minimum amount is 8192(1024 * 8)

// 5. D (Explanation not needed because its covered in the chapter)

// 6. D (Explanation not needed because its covered in the chapter)

// 7. C (Explanation not needed because its covered in the chapter)

// 8. C (Explanation not needed because its in the chapter)

// 9. 
// Number of threads in a block: 128
// Number of threads in a grid: 128 * (N + 128 - 1) = 128 * (200000 + 128 - 1)= 128 * 1563 = 200064
// Number of blocks in the grid: (200000 + 128 - 1) / 128;
// What is the number of threads that execute the code on line 02? 2000064 due to CUDA creating the amount of threads we ask it to.
// What is the number of threads that execute the code on line 04? Code on line 04 filters all the threads which shouldn't have been launched where we to launch exactly 200000 threads, 
// therefore the answer is 200000 threads because 64 threads are filtered by the if statement

// 10. __host__ __device__ indicates the function can be run on both the CPU and GPU, therefore intern doesn't need to define the function two times. Skill issue...
