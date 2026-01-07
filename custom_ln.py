"""
Simple custom layernorm to learn :)
"""

import torch
import triton
import triton.language as tl

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=["N", "BLOCK_SIZE"],
)
@triton.jit
def _layer_norm_fwd_fused(
    x_ptr,  # pointer to the input
    y_ptr,  # pointer to the output
    w_ptr,  # pointer to the weights
    b_ptr,  # pointer to the biases
    mu_ptr,  # pointer to the mean
    rstd_ptr,  # pointer to the 1/std
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    local_mask = offs < N
    irange = pid * N + offs
    x = tl.load(x_ptr + irange, local_mask, other=0.0).to(tl.float32)
    mu = tl.sum(x) / N
    tl.store(mu_ptr + pid, mu)

    x_shift = tl.where(local_mask, x - mu, 0.0)

    var = tl.sum(x_shift * x_shift) / N

    rstd = tl.rsqrt(var + eps)

    tl.store(rstd_ptr + pid, rstd)

    w = tl.load(w_ptr + offs, local_mask).to(tl.float32)
    b = tl.load(b_ptr + offs, local_mask).to(tl.float32)

    x_norm = tl.fma((x_shift * rstd), w, b)
    tl.store(y_ptr + irange, x_norm.to(OUT_DT), local_mask)


@triton.jit
def _layer_norm_bwd_fused(
    dLdy_ptr,  # pointer to dLdy
    x_ptr,  # pointer to input [M, N]
    w_ptr,  # pointer to weight [N,]
    mu_ptr,  # pointer to cached mu [M,]
    rstd_ptr,  # pointer to cached rstd [M,]
    dx_ptr,  # output pointer to output dX
    dw_ptr,  # output pointer to partial sums output for dw. [GROUP_SIZE_M, N]
    db_ptr,  # output pointer to partial sums output for db. [GROUP_SIZE_M, N]
    lock_ptr,  # pointer to db&dw locks. [GROUP_SIZE_M,]
    N,  # number of columns in X
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    lrange = tl.arange(0, BLOCK_SIZE)
    mask = lrange < N

    x = tl.load(x_ptr + pid * N + lrange, mask, other=0.0)
    w = tl.load(w_ptr + lrange, mask, other=0.0)
    mu = tl.load(mu_ptr + pid, mask, other=0.0)

    dy = tl.load(dLdy_ptr + pid * N + lrange, mask, other=0.0)

    lock_idx = pid % GROUP_SIZE_M
    db_addrs = db_ptr + lock_idx * N + lrange

    partial_db = dy.to(tl.float32)  # cast for accum
    while tl.atomic_cas(lock_ptr + lock_idx, 0, 1) != 0:
        pass

    partial_db += tl.load(db_addrs, mask, other=0.0, cache_modifier=".cg")
    tl.store(db_addrs, partial_db, mask=mask, cache_modifier=".cg")

    tl.debug_barrier()  # all threads need to have completed their work before we can free the row

    tl.atomic_xchg(lock_ptr + lock_idx, 0)


def calculate_settings(n: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = calculate_settings(N)
        _layer_norm_fwd_fused[(M,)](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mu, rstd = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        dw = torch.empty((N,), dtype=x.dtype, device=x.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)
        BLOCK_SIZE = calculate_settings(N)

        # Each program handles next_power_of_2(M / num_sms) rows = GROUP_SIZE_M (e.g 2048 / 82 = 25 => 32)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        GROUP_SIZE_M = triton.next_power_of_2(M // sm_count)
        print(f"GROUP_SIZE_M: {GROUP_SIZE_M}")

        dw_partial = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=w.device)
        db_partial = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=w.device)
        db_locks = torch.zeros((GROUP_SIZE_M,), dtype=torch.int32, device=w.device)

        _layer_norm_bwd_fused[(M,)](
            dLdy,
            x,
            w,
            mu,
            rstd,
            dLdx,
            dw_partial,
            db_partial,
            db_locks,
            N,
            GROUP_SIZE_M,
            BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )

        db = db_partial.sum(dim=0)

        return dLdx, None, dw, db, None
