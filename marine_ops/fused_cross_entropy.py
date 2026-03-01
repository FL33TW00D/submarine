"""
Final Linear + Cross Entropy

Backward is performed during the forward, as per: https://github.com/mgmalek/efficient_cross_entropy/tree/main

https://excalidraw.com/#json=1JXMLbU3fSo4LtJzIx3ue,eQpQl510IQ0BCCLRuK68SQ

I think there is some SM90 specific optimizations that can be done here via DSMEM.
"""

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


# sm86 mma:
#  A(16×16) @ B(16×8) → C(16×8)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 256, "BLOCK_K": 64, "ROW_PER_BLOCK": 64}, num_warps=16, num_stages=3),
    ],
    key=["D", "V"],
)
@triton.jit
def _cross_entropy_fwdbwd_fused(
    x_ptr,  # (BT, D)
    weight_ptr,  # (V,D)
    target_ptr,  # (BT,)
    loss_ptr,  # (ROW_PER_BLOCK,)
    ROW_PER_BLOCK: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,  # hidden dim
    V: tl.constexpr,  # Vocab size
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)  # controls which rows get solved
    bt_row_range = tl.arange(0, ROW_PER_BLOCK) + (pid * ROW_PER_BLOCK)

    target = tl.load(target_ptr + bt_row_range)

    x_bp = tl.make_block_ptr(
        x_ptr + (pid * ROW_PER_BLOCK * D),
        shape=(ROW_PER_BLOCK, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(ROW_PER_BLOCK, BLOCK_K),
        order=(1, 0),
    )

    w_bp = tl.make_block_ptr(
        weight_ptr,
        shape=(V, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_V, BLOCK_K),
        order=(1, 0),
    )

    max_logit = tl.full((ROW_PER_BLOCK,), value=float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros((ROW_PER_BLOCK,), dtype=tl.float32)
    target_logit = tl.zeros((ROW_PER_BLOCK,), dtype=tl.float32)

    for v_idx in range(0, tl.cdiv(V, BLOCK_V)):
        logits_acc = tl.zeros((ROW_PER_BLOCK, BLOCK_V), dtype=tl.float32)

        for _ in range(0, tl.cdiv(D, BLOCK_K)):
            x_sub = tl.load(x_bp)
            w_sub = tl.load(w_bp)
            logits_acc += tl.dot(x_sub, tl.trans(w_sub))
            x_bp = tl.advance(x_bp, (0, BLOCK_K))
            w_bp = tl.advance(w_bp, (0, BLOCK_K))

        x_bp = tl.advance(x_bp, (0, -D))
        w_bp = tl.advance(w_bp, (BLOCK_V, -D))

        # online softmax correction
        chunk_max = tl.max(logits_acc, axis=1)
        new_max = tl.maximum(max_logit, chunk_max)
        # rescale old sum_exp to new max
        sum_exp = sum_exp * tl.exp(max_logit - new_max)
        # add new chunk's contribution
        sum_exp += tl.sum(tl.exp(logits_acc - new_max[:, None]), axis=1)
        max_logit = new_max

        # capture target logit
        v_start = v_idx * BLOCK_V
        col_idx = target - v_start
        mask = (col_idx >= 0) & (col_idx < BLOCK_V)
        target_val = tl.sum(
            logits_acc * (tl.arange(0, BLOCK_V)[None, :] == col_idx[:, None]).to(tl.float32),
            axis=1,
        )
        target_logit += tl.where(mask, target_val, 0.0)

    loss = tl.log(sum_exp) + max_logit - target_logit
    tl.store(loss_ptr + bt_row_range, loss)


def calculate_settings(n: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class MarineLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,  # [B,T,D]
        weight: torch.Tensor,  # [V,D]
        target: torch.Tensor,  # [BT,]
        reduction: str = "mean",
    ):
        capability = torch.cuda.get_device_capability()
        _sm_version = capability[0] * 10 + capability[1]

        x_arg = x.reshape(-1, x.shape[-1])  # (B,T,D) -> (B*T,D)
        target_arg = target.reshape(-1)
        BT, D = x.shape

        V, D = weight.shape
        loss = torch.empty(BT, dtype=torch.float32, device=x.device)  # Loss per token

        # dX = torch.empty_like(x)
        # dW = torch.empty_like(weight)

        # V == 201 088 for gpt-oss 120B

        BLOCK_V = 128
        BLOCK_K = 64
        ROW_PER_BLOCK = 64
        num_warps = 8
        N_PROGRAMS = cdiv(BT, ROW_PER_BLOCK)

        _cross_entropy_fwdbwd_fused[(N_PROGRAMS,)](  #
            x_arg,
            weight,
            target_arg,
            loss,
            # ROW_PER_BLOCK=tl.constexpr(ROW_PER_BLOCK),
            # BLOCK_V=tl.constexpr(BLOCK_V),
            # BLOCK_K=tl.constexpr(BLOCK_K),
            D=tl.constexpr(D),
            V=tl.constexpr(V),
            OUT_DT=tl.constexpr(tch_to_trt[x_arg.dtype]),
            # num_warps=num_warps,
        )
        # ctx.save_for_backward(dX, dW)
        print(_cross_entropy_fwdbwd_fused.best_config)
        return loss

    @staticmethod
    def backward(ctx, dLdy: torch.Tensor):
        pass


def fused_linear_cross_entropy(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return MarineLinearCrossEntropy.apply(x, weight, target)
