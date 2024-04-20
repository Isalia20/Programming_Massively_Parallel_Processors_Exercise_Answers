//1. 
__global__ void foo_kernel(int* a, int* b){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104){
        b[i] = a[i] + 1;
    }
    if (i % 2 == 0){
        a[i] = b[i] * 2;
    }
    for (unsigned int j = 0; j < 5 - (i%3); j++){
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d){
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d);
}
// a. What is the number of warps per block
// 4 (128 / 32) given warp size of 32

// b. What is the number of warps in the grid?
// 4 * (1024 + 128 - 1) / 128;

// c. 
// First we find out how many threads are running 
// i. How many warps in the grid are active?
// 8 blocks * 128 threads per block = 1024. PERFECT!!!
// 960 threads are active on line 4, therefore 960 / 32=30 warps
// ii. How many warps in the grid are divergent?
// Let's calculate, 0 to 31 threads all execute the line 04
// for threads from 32 to 63. 32 to 39 execute the statement, while 
// 40 to 63 do not therefore this warp is divergent. 
// 64 to 95 do not execute at all so warp is not divergent.
// 96 to 127 is divergent because threads 96 to 103 do not execute
// while 104+ threads all execute.
// Ans: 2 warps are divergent
// iii. What is the SIMD efficiency in % of warp 0 of block 0?
// 100% because all threads are executing.
// iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
// Since only 8 threads are executing we get efficiency of:
// 8 / 32 = 25%
// v. What is the SIMD efficiency (in %) of warp 3 of block 0?
// Since 24 threads are active
// 24 / 32 = 75%

// d. For the statement on line 07
// i. How many warps in the grid are active?
// All the warps(32) are active because all the warps contain threadIds which are even
// ii. How many warps in the grid are divergent?
// All 32 warps will be divergent due to them containing both even and odd threadIdxs
// iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
// 50%

// e. For the loop on line 09
// i. How many iterations have no divergence? 
// ii. How many iterations have divergence
// 5 - i % 3
// i % 3 can be -> 0, 1, 2
// so the iteration can go up to 3, 4 or 5
// Since in a warp we will always get values which
// return 0, 1, 2 then some iterations will go to 3, 4 or 5
// Therefore all the threads in all warps will have divergence
// For iterations, iterations that go up to 3 or 4 will have divergence
// due to them waiting till 5


// 2.
// For a vector addition, assume that the vector length is 2000, each thread
// calculates one output element, and the thread block size is 512 threads. How
// many threads will be in the grid?
// 2048 threads will be launched

// 3. 
// For the previous question, how many warps do you expect to have divergence
// due to the boundary check on vector length?
// 48 threads will not be executed, last 32 threads won't be executed at all so they don't have divergence
// now we are left with 2016 threads, 2000 threads do execute, 16 do not, last warp will contain 32 threads 
// out of which 16 threads will be inactive, therefore 1 warp will have divergence

// 4. 
// Consider a hypothetical block with 8 threads executing a section of code
// before reaching a barrier. The threads require the following amount of time
// (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and
// 2.9; they spend the rest of their time waiting for the barrier. What percentage
// of the threads’ total execution time is spent waiting for the barrier?
// Find the maximum of the microseconds, which is 3.0.
// Thread 1: 3.0 - 2.0 = 1.0 microseconds
// Thread 2: 3.0 - 2.3 = 0.7 microseconds
// Thread 3: 3.0 - 3.0 = 0.0 microseconds
// Thread 4: 3.0 - 2.8 = 0.2 microseconds
// Thread 5: 3.0 - 2.4 = 0.6 microseconds
// Thread 6: 3.0 - 1.9 = 1.1 microseconds
// Thread 7: 3.0 - 2.6 = 0.4 microseconds
// Thread 8: 3.0 - 2.9 = 0.1 microseconds
// Total time spent waiting: 4.1 microseconds
// total execution time: 19.9
// 4.1 / 19.9 
// for each thread


// 5.
// A CUDA programmer says that if they launch a kernel with only 32 threads
// in each block, they can leave out the __syncthreads() instruction wherever
// barrier synchronization is needed. Do you think this is a good idea? Explain.
// It's not a good idea to leave syncthreads.
// One of the reasons is that while it might work on the current GPUs
// warp size might be different for the future GPUs or older ones
// therefore relying on warp size staying 32 is a bad idea
// plus whenever someone else is reading the code, it might not be 
// clear that there is a synchronization happening at some point
// because kernel is launched with 32 threads in each block

// 6.
// If a CUDA device’s SM can take up to 1536 threads and up to 4 thread
// blocks, which of the following block configurations would result in the most
// number of threads in the SM?
// a. 128 threads per block
// b. 256 threads per block
// c. 512 threads per block
// d. 1024 threads per block
// C, but we should use 3 blocks and not 4 thread block
// to get the maximum occupancy

// 7. 
// Assume a device that allows up to 64 blocks per SM and 2048 threads per
// SM. Indicate which of the following assignments per SM are possible. In the
// cases in which it is possible, indicate the occupancy level.
// a. 8 blocks with 128 threads each - Possible 50%
// b. 16 blocks with 64 threads each - Possible 50%
// c. 32 blocks with 32 threads each - Possible 50%
// d. 64 blocks with 32 threads each - Possible 100%
// e. 32 blocks with 64 threads each - Possible 100%


// 8. 
// Consider a GPU with the following hardware limits: 2048 threads per SM, 32
// blocks per SM, and 64K (65,536) registers per SM. For each of the following
// kernel characteristics, specify whether the kernel can achieve full occupancy.
// If not, specify the limiting factor.
// a. The kernel uses 128 threads per block and 30 registers per thread.
// b. The kernel uses 32 threads per block and 29 registers per thread.
// c. The kernel uses 256 threads per block and 34 registers per thread.

// a. 
// 2048 threads per SM, with 128 threads per blocks, means 16 blocks max per SM
// 26 blocks max per SM means we will occupy:
// 128 * 30 * 16= 61440
// 61440/65536 = 93% occupancy
// problem is the amount of threads here

// b.
// maximum amount of blocks we can create with 32 threads is 64, but our SM is limited to 32 blocks per SM
// therefore we would have 32 * 32 * 29 = 29696 
// which is a very low occupancy. Problem here is low amount of thread count per block


// c.
// maximum amount of blocks we can create with 256 threads is 8. 
// 8 * 256 * 34 is 69632. 
// We achieved full occupancy but amount of registers isn't enough, therefore we can't have this configuration
// because 69k > 65k

// 9.
// student mentions that they were able to multiply two 1024 x 1024 matrices
// using a matrix multiplication kernel with 32 x 32 thread blocks. The student is
// using a CUDA device that allows up to 512 threads per block and up to 8 blocks
// per SM. The student further mentions that each thread in a thread block calculates
// one element of the result matrix. What would be your reaction and why?

// How were they able to do this. If CUDA device allows up to 512 threads per block and up to 8 blocks
// with 32 * 32, we get 1024 threads per block. 512 < 1024, so not sure how student launched that kernel
