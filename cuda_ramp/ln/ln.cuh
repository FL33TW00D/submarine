#include <cuda_bf16.h>

#define VPT 8

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __device__ __forceinline__ float thread_reduce_sum(const __nv_bfloat16* input, int BLOCK_SIZE, int N) {
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    int step = 128 / sizeof(__nv_bfloat16);
    float sum = 0.0f;
    for (int idx = tid * step; idx < N; idx += BLOCK_SIZE * step) {
        uint4 raw = reinterpret_cast<const uint4*>(&input[row * N + idx])[0];
        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
        float sum_local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f2 = __bfloat1622float2(pairs[i]);
            sum_local += f2.x + f2.y;
        }
        sum += sum_local;
    }
    return sum;
}
