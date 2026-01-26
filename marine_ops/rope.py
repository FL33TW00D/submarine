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


# Each program is responsible for a row


@triton.jit
def _rope_fwd(
    q_ptr,
    k_ptr,
    sin_ptr,
    cos_ptr,
    qo_ptr,
    ko_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    dr = tl.arange(0, D)
    gr = pid * D + dr
    half_D = D // 2

    q = tl.load(q_ptr + gr)
    k = tl.load(k_ptr + gr)

    seq_pos = pid % S
    sc_offset = seq_pos * D + dr
    sin = tl.load(sin_ptr + sc_offset)
    cos = tl.load(cos_ptr + sc_offset)

    partner_local = tl.where(dr < half_D, dr + half_D, dr - half_D)
    partner_idx = pid * D + partner_local
    sign = tl.where(dr < half_D, -1.0, 1.0)

    q_rotated = tl.load(q_ptr + partner_idx) * sign
    k_rotated = tl.load(k_ptr + partner_idx) * sign

    q_embed = q * cos + q_rotated * sin
    k_embed = k * cos + k_rotated * sin

    tl.store(qo_ptr + gr, q_embed)
    tl.store(ko_ptr + gr, k_embed)


class MarineRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, sin_cached: torch.Tensor, cos_cached: torch.Tensor):
        (B, S, D) = q.shape
        assert (D & (D - 1) == 0) and D != 0  # power of 2

        M = B * S

        q = q.reshape(-1, D)
        k = k.reshape(-1, D)

        qo = torch.empty_like(q)
        ko = torch.empty_like(k)

        _rope_fwd[(M,)](q, k, sin_cached, cos_cached, qo, ko, B, S, D)

        qo = qo.reshape(B, S, D)
        ko = ko.reshape(B, S, D)

        ctx.save_for_backward(sin_cached, cos_cached)
        return qo, ko

    @staticmethod
    def backward(ctx, dqo, dko):
        sin_cached, cos_cached = ctx.saved_tensors
        dq, dk = MarineRoPE.apply(dqo, dko, -sin_cached, cos_cached)
        return dq, dk, None, None
