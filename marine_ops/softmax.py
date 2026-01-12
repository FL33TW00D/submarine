from typing import Optional
import torch
import triton
import triton.language as tl

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _softmax_fwd_fused(
    x_ptr,  # pointer to the input
    y_ptr,  # pointer to the output
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    row_start = pid * N
    x = tl.load(x_ptr + row_start + offs, mask, other=0.0)
    # One pass over data for max
    x_max = tl.max(x)  # scalar maximum for the row

    x_shift = x - x_max
    x_shift = tl.where(offs < N, x_shift, float("-inf"))
    x_exp = tl.exp(x_shift)
    # One pass over data for denom
    denom = tl.sum(x_exp)

    # One pass over data for division
    y = x_exp / denom
    tl.store(y_ptr + row_start + offs, y, mask)


"""
Can be derived with quotient rule and some effort.
See: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
And: https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function

def softmax_backward(dLdy, y):
    dot = (dLdy * y).sum(dim=-1, keepdim=True)
    
    # Final gradient: sigma * (g - dot)
    return y * (dLdy - dot)

"""


@triton.jit
def _softmax_bwd_fused(
    dLdy_ptr,  # pointer to dLdy (upstream gradient) [M,N]
    y_ptr,  # pointer to the output of softmax [M,N]
    dLdx_ptr,  # pointer to the result of this backward [M,N]
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    lrange = tl.arange(0, BLOCK_SIZE)
    mask = lrange < N
    row_offset = pid * N

    dy = tl.load(dLdy_ptr + row_offset + lrange, mask, other=0.0)
    y = tl.load(y_ptr + row_offset + lrange, mask, other=0.0)

    prod = dy * y  # elementwise mul
    red = tl.sum(prod, axis=0)

    dLdx = y * (dy - red)
    tl.store(dLdx_ptr + row_offset + lrange, dLdx, mask)


def calculate_settings(n: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class MarineSoftmax(torch.autograd.Function):
    """
    torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: Optional[int] = None):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        BLOCK_SIZE = calculate_settings(N)
        _softmax_fwd_fused[(M,)](  #
            x_arg,
            y,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dLdy: torch.Tensor):
        (y,) = ctx.saved_tensors
        y = y.reshape(-1, y.shape[-1])
        M, N = y.shape

        dLdx = torch.empty_like(dLdy)
        BLOCK_SIZE = calculate_settings(N)

        _softmax_bwd_fused[(M,)](dLdy, y, dLdx, N, BLOCK_SIZE, OUT_DT=tch_to_trt[y.dtype])

        return dLdx, None
