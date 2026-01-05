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


@triton.jit
def _layer_norm_fwd_fused(
    x_ptr,  # pointer to the input
    y_ptr,  # pointer to the output
    w_ptr,  # pointer to the weights
    b_ptr,  # pointer to the biases
    mu_ptr,  # pointer to the mean
    rstd_ptr,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
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

    x_shift = tl.where(local_mask, x - mu, 0.0)

    var = tl.sum(x_shift * x_shift) / N

    w = tl.load(w_ptr + offs, local_mask).to(tl.float32)
    b = tl.load(b_ptr + offs, local_mask).to(tl.float32)

    x_norm = w * (x_shift * tl.rsqrt(var + eps)) + b
    tl.store(y_ptr + irange, x_norm.to(OUT_DT), local_mask)


class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        _layer_norm_fwd_fused[(M,)](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            x_arg.stride(0),
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return y
