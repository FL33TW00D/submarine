#define VPT 8  // D / BLOCK_SIZE = 2048 / 256

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float thread_reduce_sum(const __nv_bfloat16* input, int BLOCK_SIZE, int N) {
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


__global__ void ln_kernel(const __nv_bfloat16* input, __nv_bfloat16* output, float eps, int N) {
    extern __shared__ float shmem[];
    const int tid = threadIdx.x;
    const int BLOCK_SIZE = blockDim.x;
    const int row = blockIdx.y;

    float sum = thread_reduce_sum(input, BLOCK_SIZE, N);
    sum = warp_reduce_sum(sum);

    int warp_id = tid / warpSize;
    if (tid % warpSize == 0) {
        shmem[warp_id] = sum;
    }
    __syncthreads();

    int num_warps = BLOCK_SIZE / warpSize;
    if (warp_id == 0 && tid < num_warps) {
        float val = shmem[tid];
        for (int offset = num_warps / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0)
            shmem[0] = val;
    }
    __syncthreads();

    sum = shmem[0];
    float mu = sum / N;

    float x_shift[VPT];
    float sq_shift_sum = 0.0f;
    int count = 0;

    int step = 128 / sizeof(__nv_bfloat16);
    for (int idx = tid * step; idx < N; idx += BLOCK_SIZE * step) {
        uint4 raw = reinterpret_cast<const uint4*>(&input[row * N + idx])[0];
        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
         #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f2 = __bfloat1622float2(pairs[i]);
            float xs = f2.x - mu;
            x_shift[count++] = xs;
            sq_shift_sum += xs * xs;

            float ys = f2.y - mu;
            x_shift[count++] = ys;
            sq_shift_sum += ys * ys;
        }
    }

    sq_shift_sum = warp_reduce_sum(sq_shift_sum);
    if (tid % warpSize == 0) {
        shmem[warp_id] = sq_shift_sum;
    }
    __syncthreads();

    if (warp_id == 0 && tid < num_warps) {
        float val = shmem[tid];
        for (int offset = num_warps / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0)
            shmem[0] = val;
    }
    __syncthreads();

    float var = shmem[0] / N;
    float rstd = rsqrtf(var + eps);

    count = 0;
    for (int idx = tid * step; idx < N; idx += BLOCK_SIZE * step) {
        __nv_bfloat162 out_pairs[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float nx = x_shift[count++] * rstd;
            float ny = x_shift[count++] * rstd;
            out_pairs[i] = __floats2bfloat162_rn(nx, ny);
        }
        reinterpret_cast<uint4*>(&output[row * N + idx])[0] =
            reinterpret_cast<uint4*>(out_pairs)[0];
    }
}
