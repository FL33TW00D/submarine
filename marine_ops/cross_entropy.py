"""
Testing ground for xentropy kernels.

sm_90 >= can use DSMEM for large reductions
sm_90 < uses standard blocked approach

"""

from typing import Optional
import torch
import triton
import triton.language as tl


tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def cdiv(n, d):
    return -(n // -d)


"""
Forward:
$$-z_t + log(\\Sigma{e^z_t})$$

# Formula falls out easily from H(P,Q) = xentropy, consider the onehot
"""


@triton.jit
def _cross_entropy_fwd_fused(
    logits_ptr,  # pointer to the logits
    target_ptr,  # pointer to target class labels (BT,)
    loss_ptr,  # pointer to the
    N,  # number of columns (vocab size)
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_offset = pid * N
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < N

    target = tl.load(target_ptr + pid)
    z_t = tl.load(logits_ptr + row_offset + target)

    row_sum_exp = 0.0
    running_max = float("-inf")
    for k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        chunk = tl.load(logits_ptr + row_offset + cols, col_mask, 0.0)
        cur_max = tl.max(chunk)
        new_max = tl.maximum(cur_max, running_max)

        row_sum_exp = row_sum_exp * tl.exp(running_max - cur_max) + tl.sum(tl.exp(chunk - new_max))
        running_max = new_max

    result = tl.log(row_sum_exp) + running_max - z_t
    tl.store(loss_ptr + pid, result)


"""
$$softmax(\textbf{z})_i-{\\delta}_{ij}$$

-z_t -> kronecker
-d/dx[log(x)] -> 1/x
-d/dx[log(sigma(e^z_t))] -> 1/S . dS/dx. d/dx[sigma(e^z_t)] only matters on matching index, so is e^z_t
- therefore, it's softmax! 
"""


@triton.jit
def _cross_entropy_bwd_fused(
    dLdy_ptr,  # pointer to dLdy (upstream gradient) [M,N]
    y_ptr,  # pointer to the output of softmax [M,N]
    dLdx_ptr,  # pointer to the result of this backward [M,N]
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
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


class MarineCrossEntropy(torch.autograd.Function):
    """
    Usage:
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
        logits_arg = logits.reshape(-1, logits.shape[-1])  # (B,T,V) -> (B*T,V)
        target_arg = target.reshape(-1)
        BT, V = logits_arg.shape
        loss = torch.empty(BT, dtype=torch.float32, device=logits.device)  # Loss per token
        # Each program solves a row, by looping the required number of times
        # V == 201 088 for gpt-oss 120B
        BLOCK_SIZE = 16384  # TODO: autotoon
        # 201 088 / 16384 == 13 loops
        _cross_entropy_fwd_fused[(BT,)](  #
            logits_arg,
            target_arg,
            loss,
            V,
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            OUT_DT=tl.constexpr(tch_to_trt[logits_arg.dtype]),
        )
        # ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, dLdy: torch.Tensor):
        return None * 3
