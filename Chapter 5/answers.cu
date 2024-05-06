// 1. Consider matrix addition. Can one use shared memory to reduce the
// global memory bandwidth consumption? Hint: Analyze the elements that
// are accessed by each thread and see whether there is any commonality
// between threads.
// Answer:
// Matrix addition won't benefit from using shared memory because of an easy reason
// multiple threads won't be accessing the same elements either from M or N matrix.
// One number is used exactly once for addition so shared memory won't give us any benefit

// 2. NA

// 3. What type of incorrect execution behavior can happen if one forgot to use
// one or both __syncthreads() in the kernel of Fig. 5.9?
// Answer:
// If one forgets to call syncthreads whenever we are using shared memory. Some of the threads
// might have already written to the location and might proceed to the next step of the algorithm
// (matmul lets say), during this time some of the threads who haven't finished writing in X memory location,
// that X memory location might be read by the thread that has already finished writing, therefore producing
// incorrect output.

// 4. Assuming that capacity is not an issue for registers or shared memory, give
// one important reason why it would be valuable to use shared memory
// instead of registers to hold values fetched from global memory? Explain
// your answer.
// Answer:
// If capacity is not an issue for registers or shared memory, using shared memory is still 
// valuable because we still have to access the elements from global memory. So instead 
// of accessing global memory X times, we can access it 1 time and use the shared memory
// rather than accessing it multiple times.

// 5. For our tiled matrix-matrix multiplication kernel, if we use a 32 x 32 tile,
// what is the reduction of memory bandwidth usage for input matrices M
// and N?
// Answer:
// For our tiled matmul kernel using 32x32 tile will result in reduction of memory bandwidth by 32

// 6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of
// which has 512 threads. If a variable is declared as a local variable in the
// kernel, how many versions of the variable will be created through the
// lifetime of the execution of the kernel?
// Answer:
// Since variable will be created by each thread privately, it will be 512,000 versions of the variable

// 7. In the previous question, if a variable is declared as a shared memory
// variable, how many versions of the variable will be created through the
// lifetime of the execution of the kernel?
// Answer:
// 1000 versions of variables, since blocks do not share memory but threads inside of a block share the memory

// 8. Consider performing a matrix multiplication of two input matrices with
// dimensions N x N. How many times is each element in the input matrices
// requested from global memory when:
// a. There is no tiling?
// b. Tiles of size T x T are used?
// Answer:
// a. Each element is requested N times if there is no tiling
// b. Each element is requested N/T times if there is tiling.

// 9.A kernel performs 36 floating-point operations and seven 32-bit global
// memory accesses per thread. For each of the following device
// properties, indicate whether this kernel is compute-bound or memory-
// bound.
// a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second
// b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second
// Answer:
// a. 36 FLOP per thread and 7 global memory access per thread
// let's test the maximum amount of threads that can be launched considering max flop count
// 200 * 10^9 flops can be done. If we have 36 FLOPs per thread, that means we can have maximum of
// 5 555 555 555 threads launched together. If we have 7 global memory access for which we have 4 bytes of float
// per access to be transferred, then we have 28 bytes per thread to transfer. 28 * 5 555 555 555 threads 
// is 155GB in total which is more than peak memory bandwidth 100GB/second, therefore we are memory bound
// b. Same calculations but replace the GFLOPS and peak memory bandwidth, turns out to 233 GB/second 
// which means we are compute bound since 250 > 233

//10.
// To manipulate tiles, a new CUDA programmer has written a device kernel
// that will transpose each tile in a matrix. The tiles are of size
// BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of
// matrix A is known to be a multiple of BLOCK_WIDTH. The kernel
// invocation and code are shown below. BLOCK_WIDTH is known at
// compile time and could be set anywhere from 1 to 20.

#define BLOCKWIDTH 3
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void 
BlockTranspose(float* A_elements, int A_width, int A_height){
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

// A. Out of the possible range of values for BLOCK_SIZE, for what values
// of BLOCK_SIZE will this kernel function execute correctly on the
// device?
// B. If the code does not execute correctly for all BLOCK_SIZE values, what
// is the root cause of this incorrect execution behavior? Suggest a fix to the
// code to make it work for all BLOCK_SIZE values
// Answer:
// A. it will work for block_size = 1 since then we won't need to do syncthreads.
// for that blocksize sync threads is not needed, however for a larger block size 
// syncthreads is needed to make sure that all threads have copied in the blockA
// variable from A_elements variable
// B. Root cause will be missing syncthreads. after blockA[threadIdx.y][threadIdx.x]

// 11. 
// Consider the following CUDA kernel and the corresponding host function
// that calls it:
__global__ void foo_kernel(float* a, float* b){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for (unsigned int j = 0; j < 4; j++){
        x[j] = a[j * blockDim.x * gridDim.x + i];
    }
    if (threadIdx.x == 0){
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3] + ys * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
}
void foo(int* a_d, int* b_d){
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1) / 128, 128 >>>(a_d, b_d);
}
//How many versions of the variable i are there?
// b. How many versions of the array x[] are there?
// c. How many versions of the variable y_s are there?
// d. How many versions of the array b_s[] are there?
// 120 CHAPTER 5 Memory architecture and data locality
// e. What is the amount of shared memory used per block (in bytes)?
// f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?
// Answer:
// a. As many threads there are that many i's are there(8 * 128)
// b. As many threads there are that many x[4] will be available( 8 * 128);
// c. As many blocks there are available that many y_s(8 in this case);
// d. As many blocks there are available that many b_s(8);
// e. 4 bytes + 128 * 4 bytes = 516 bytes per block
// f. 12 floating point operations and 5 global memory accesses(one for a and another one for b), meaning (12 / 5) OP/B
// 4 accesses from a in a for loop and 1 access from b when copying to b_s

// 12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32
// blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM.
// For each of the following kernel characteristics, specify whether the kernel
// can achieve full occupancy. If not, specify the limiting factor.
// a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared
// memory/SM.
// b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of
// shared memory/SM.

// Answer:
// a. 64 threads, 27 registers and 4 kb of shared memory/SM. 
// 1728 registers per block, maximum of 37(65536/1728) blocks can be launched but since we are limited by block count
// we will be able to launch only 32 blocks. If we launch 32 blocks, 32(blocks) * 64(threads) * 27(registers) = 55,296 registers
// it achieves 84% occupancy(55296/65536). Now let's check shared memory
// 4 * 32 = 128 KB for shared memory due to launching 32 blocks, therefore we can't achieve full occupancy because we will exceed
// the shared memory limit.

// b. 256 threads, 31 registers and 8 kb of shared memory.
// 7936 registers per block, maximum of 8 blocks can be launched which fits into the maximum
// amount of blocks that can be launched. Now let's check shared memory
// 96 KB of shared memory, if we launch 8 blocks with 8 kb of shared memory we get
// 64 KB of shared memory which fits in the requirements.
// 96.8% occupancy can be achieved