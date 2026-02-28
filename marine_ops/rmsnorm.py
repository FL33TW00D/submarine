from typing import Optional, Self

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from hypothesis import given, settings, strategies as st

DEVICE = triton.runtime.driver.active.get_active_torch_device()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _rms_norm_fwd_fused(
    x_ptr,  # pointer to the input [M,N]
    w_ptr,  # pointer to weight [N,]
    rrms_ptr,  # pointer to 1/RMS [M,]
    y_ptr,  # pointer to the output [M,N]
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    # Load X
    # Load M
    pid = tl.program_id(0)
    offset = pid * N

    rnge = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offset + rnge, rnge < N, 0.0).to(tl.float32)
    w = tl.load(w_ptr + rnge, rnge < N).to(tl.float32)

    mu_sq = tl.sum((x * x).cast(tl.float32)) * (1 / N)
    rrms = tl.rsqrt(mu_sq + eps)

    tl.store(rrms_ptr + pid, rrms)

    y = rrms * x * w
    tl.store(y_ptr + offset + rnge, y.to(OUT_DT), rnge < N)


"""
Derive using quotient rule.
Uses same associativity trick as softmax to avoid materializing full jacobian.
"""


@triton.jit
def _rms_norm_bwd_fused(
    dLdy_ptr,  # pointer to dLdy [M,N]
    dLdx_ptr,  # pointer to dLdx [M,N]
    w_ptr,  # pointer to weight [N,]
    rrms_ptr,  # pointer to recip rms [M,]
    y_ptr,  # pointer to xhat [M,N] TODO: recompute
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    rnge = tl.arange(0, BLOCK_SIZE)
    offset = pid * N
    rmask = rnge < N

    offset = pid * N
    dy = tl.load(dLdy_ptr + offset + rnge, rmask, 0.0)
    y = tl.load(y_ptr + offset + rnge, rmask, 0.0)
    w = tl.load(w_ptr + rnge, rmask, 0.0)
    rrms = tl.load(rrms_ptr + pid)

    # compute v ⊙ w
    vw = dy * w
    dx = rrms * (vw - (tl.sum(vw * y) / N) * y)
    tl.store(dLdx_ptr + offset + rnge, dx, rmask)


def calculate_settings(n: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class MarineRMSNorm(torch.autograd.Function):
    TOLS = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 7e-2}

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rrms = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = calculate_settings(N)
        _rms_norm_fwd_fused[(M,)](
            x_arg,
            weight,
            rrms,
            y,
            N,
            eps,
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            OUT_DT=tl.constexpr(tch_to_trt[x.dtype]),
        )
        ctx.save_for_backward(weight, rrms, y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        weight, rrms, y = ctx.saved_tensors

        M, N = y.shape
        print(f"M: {M}, N:{N}")
        dLdx = torch.empty_like(dLdy)

        BLOCK_SIZE = calculate_settings(N)
        _rms_norm_bwd_fused[(M,)](
            dLdy,
            dLdx,
            weight,
            rrms,
            y,
            N,
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            OUT_DT=tl.constexpr(tch_to_trt[y.dtype]),
        )
        return dLdx, None, None, None, None
