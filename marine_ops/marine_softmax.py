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
    # 3 "loops"
    # 1. Compute x_max
    # 2. Compute denom = tl.sum(exp(x - x_max))
    # 3. Compute out = x / denom
    # each program handles 1 row

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    row_start = pid * N
    x = tl.load(x_ptr + row_start + offs, mask, other=0.0)
    x_max = tl.max(x)  # scalar maximum for the row
    # second loop, compute denom
    x_shift = x - x_max
    x_shift = tl.where(offs < N, x_shift, float("-inf"))

    x_exp = tl.exp(x_shift)
    denom = tl.sum(x_exp)
    y = x_exp / denom
    tl.store(y_ptr + row_start + offs, y, mask)


@triton.jit
def _softmax_bwd_fused():
    pid = tl.program_id(0)
    pass


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
        ctx.save_for_backward(x, y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dLdy: torch.Tensor):
        pass
