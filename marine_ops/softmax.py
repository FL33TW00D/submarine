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
    # Each program handles a single row
    # Load from x
    # compute max
    # subtract max through
    # compute sum of all
    pid = tl.program_id(0)

    offset = pid * N
    rnge = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset + rnge, rnge < N, 0.0)

    x_max = tl.max(x, axis=-1)
    x_shift = tl.exp(x - x_max)
    x_shift = tl.where(rnge < N, x_shift, float("-inf"))

    denom = tl.sum(x_shift)

    x_shift /= denom

    tl.store(y_ptr + offset + rnge, x_shift, rnge < N)


"""
Can be derived with quotient rule and some effort.
See: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

def softmax_backward(dLdy, y):
    dLdy: upstream gradient (B, T, n)
    y:    softmax output    (B, T, n)

    The Jacobian of softmax is: diag(s) - ss^T
    This is diagonal + rank-1, so the VJP v^T J simplifies:

        v ⊙ s - (v·s) s        ← two terms, all O(n)
      = s ⊙ (v - (v·s))        ← factored form used below

    No n×n matrix is ever formed.


    dot = (dLdy * y).sum(dim=-1, keepdim=True)  # v·s per position: (B, T, 1)
    return y * (dLdy - dot)                       # s ⊙ (v - v·s):   (B, T, n)
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
    # Load dLdy
    # Load y
    pid = tl.program_id(0)
    rnge = tl.arange(0, BLOCK_SIZE)
    offset = pid * N

    dLdy = tl.load(dLdy_ptr + offset + rnge, rnge < N, 0.0)
    y = tl.load(y_ptr + offset + rnge, rnge < N, 0.0)

    out = y * (dLdy - tl.sum(dLdy * y, axis=-1, keep_dims=True))
    tl.store(dLdx_ptr + offset + rnge, out, rnge < N)


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
