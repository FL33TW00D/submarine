extern "C"
__global__
void softmax_kernel(const float* input, float* output, int N) {
    int row = blockIdx.x;
    input += row * N;
    output += row * N;

    // Warp reduction is butterfly, each thread works together to pass it along to his bros
    // Sequential threads need to access sequential memory addresses

    // T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
    
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, input[i]);
    
    // So this thread now has the maximum of 4 values, and we can use shuffle down sync to get the full max
    
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    //last one here broadcasts reduced value to all threads in the warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    //So now compute exp of value
    float denom = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        denom += __expf(input[i] - max_val);

    for (int offset = 16; offset > 0; offset >>= 1)
        denom += __shfl_down_sync(0xffffffff, denom, offset);

    denom = __shfl_sync(0xffffffff, denom, 0);

    //now we have the denom
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        output[i] = __expf(input[i] - max_val) / denom;
}
    






