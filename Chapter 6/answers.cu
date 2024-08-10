// 1. Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.
// Ans: See matmul_kernel.cu

// 2. For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)
// Ans: I'm not really sure about this so I will avoid answering this question

// 3. Consider the following CUDA kernel:
// For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing is not applicable:
// a. Q: The access to array a of line 05
// a. Ans: coalescing is applicable because a is an array in global memory and contiguous threads are accessing it
// b. The access to array a_s of line 05
// b. Ans: coalescing is not applicable since a_s is in shared memory
// c. The access to array b of line 07
// c. Ans: Coalescing is applicable here since contiguous threads will be accessing contiguous memory(in global memory pool)
// d. The access to array c of line 07
// d. Ans: Here coalesced access is not applicable since tx=0 tx=1 tx=2 will be accessing different parts of memory
// e. The access to array bc_s of line 07
// e. Ans: It's shared memory so coalesced access is not applicable
// f. The access to array a_s of line 10 g. The access to array d of line 10
// f. Ans: a_s not applicable, access to d is applicable because tx=0 tx=1 will be just mapped to 8, 9, 10 and will lead to coalescing
// h. The access to array bc_s of line 11
// h. Ans: again shared memory so coalesced access is not applicable
// i. The access to array e of line 11
// i. Ans: No coalescing will be done here due to threads not accessing contiguous elements

// 4. What is the floating point to global memory access ratio (in OP/B) of each of
// the following matrix-matrix multiplication kernels?
// a. The simple kernel described in Chapter 3, Multidimensional Grids and
// Data, without any optimizations applied.
// b. The kernel described in Chapter 5, Memory Architecture and Data
// Locality, with shared memory tiling applied using a tile size of 32 x 32.
// c. The kernel described in this chapter with shared memory tiling applied
// using a tile size of 32 3 32 and thread coarsening applied using a coarsening factor of 4.

// Ans:
// a. Each element from matrix A from global memory will be accessed width times(width of the 2nd matrix).
// So to think of how many times we will access global memory, is (width + width + 1) for a single output element(1 being the global memory access of output array and the other two being the memory accesses of the A and B matrices)
// Now to generalize it for all output elements, we will have (width * width * (2 * width + 1))
// For flops, per each output element we need width multiplication and width - 1 addition. so (width + (width - 1)) FLOPS
// so to calculate the answer. (2 * width - 1)
// now to answer the question we need (2 * width - 1) / (width * width * 2 * width + 1)) which simplifies to
// (2 * width - 1) / (2 * width ^ 3 + 1) 
// so we see how inefficient this kernel was

// b. Now let's apply the same thinking for the tiled matmul. Here the difference is that we dont access the global memory of A and B tensors as many times.
// Since each block in the kernel works for a single TILE in C(output) matrix then we can conclude that matrix A's each element is only accessed the amount of tiles
// which can fit in the original matrices. So if a matrix is of shape 48x48 and we have 16x16 tiles, we can say that matrix A's each element from GLOBAL memory is only accessed
// 48 / 16 = 3 times. Now to generalize, using 32x32. We have:
// (width / 32) global memory accesses per element. (width / 32 + width / 32 + 1), again 1 access for writing the output element.
// but we have left out an important detail here, coalesced memory access. Since we are accessing multiple elements at once from matrix A we will have coalesced access so instead of 
// 32 we will have 1 coalesced access. (width / 32 + width / 32 + 1) still applies but now what should we multiply this on?
// Given that we have width elements in a row, we can say that all of them will be done in a single access. So for each tile we will do only (width * (width / 32 + width / 32 + 1)) * (width / 32 * width / 32) accesses,
// the last (width / 32 * width / 32) being how many tiles there are
// Flops will be the same since we don't apply any tricks here and just do regular matmul.
// (2 * width - 1) FLOPS
// Ans: (2 * width - 1) / ((width * (width / 32 + width / 32) + 1) * (width / 32 * width / 32)). Probably something is wrong, but it's middle of August and its too hot to think

// c. ...
