from typing import Tuple
import torch
import triton
import triton.language as tl
import math

DEVICE = triton.runtime.driver.active.get_active_torch_device()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _rope_fwd(
    q_ptr,  # input q [BSD]
    k_ptr,  # input k [BSD]
    sin_ptr,  # input sin_cached [MD]
    cos_ptr,  # input cos_cached [MD]
    qrot_ptr,  # output q [BSD]
    krot_ptr,  # output k [BSD]
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    b_offset = pid_0 * S * D
    d_half = D // 2

    qk_rows = tl.arange(0, S)[:, None] * D
    qk_cols = tl.arange(0, D)
    qk_addrs = qk_rows + qk_cols
    q = tl.load(q_ptr + b_offset + qk_addrs)
    k = tl.load(k_ptr + b_offset + qk_addrs)

    rot_addrs = qk_rows + tl.where(qk_cols < d_half, qk_cols + d_half, qk_cols - d_half)
    sign = tl.where(qk_cols < d_half, -1.0, 1.0)

    qrot = tl.load(q_ptr + b_offset + rot_addrs) * sign
    krot = tl.load(k_ptr + b_offset + rot_addrs) * sign

    sin = tl.load(sin_ptr + qk_addrs)
    cos = tl.load(cos_ptr + qk_addrs)

    q_embed = (q * cos) + (qrot * sin)
    k_embed = (k * cos) + (krot * sin)

    # write them out
    tl.store(qrot_ptr + b_offset + qk_addrs, q_embed)
    tl.store(krot_ptr + b_offset + qk_addrs, k_embed)


class MarineRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, sin_cached: torch.Tensor, cos_cached: torch.Tensor):
        if q.shape[-1] % 2 != 0 or k.shape[-1] % 2 != 0:
            raise ValueError("Dimension must be even.")

        B, S, D = q.shape

        qrot = torch.empty_like(q)
        krot = torch.empty_like(k)

        _rope_fwd[(B,)](q, k, sin_cached, cos_cached, qrot, krot, B, S, D)

        return qrot, krot

    @staticmethod
    def backward(ctx, dqo, dko):
        sin_cached, cos_cached = ctx.saved_tensors
        dq, dk = MarineRoPE.apply(dqo, dko, -sin_cached, cos_cached)
        return dq, dk, None, None
