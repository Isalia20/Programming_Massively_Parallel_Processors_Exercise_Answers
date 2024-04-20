void incorrect_barrier_example(int n){
    // ...
    if (threadIdx.x % 2 == 0){
        //...
        __syncthreads{};
    } else {
        __syncthreads{};
    }
}