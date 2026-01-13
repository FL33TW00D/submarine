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
def _fa1_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    dn_ptr,
    max_ptr,
    sf,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    Br: tl.constexpr,
    Bc: tl.constexpr,
    Bcp: tl.constexpr,
    Tr: tl.constexpr,
    Tc: tl.constexpr,
):
    # Each program is responsible for solving one head. So a whole Q,K,V mat
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # TODO: Dq != Dv
    b_offset = NH * S * D  # batch offset
    h_offset = S * D  # head offset

    global_offset = pid_b * b_offset + pid_h * h_offset

    q_ptr += global_offset
    k_ptr += global_offset
    v_ptr += global_offset
    o_ptr += global_offset
    dn_ptr += pid_b * NH * S + pid_h * S
    max_ptr += pid_b * NH * S + pid_h * S

    q_stride = Br * D
    k_stride = Bc * D
    v_stride = Bc * D
    o_stride = Br * D

    kv_row_idx = tl.arange(0, Bcp)  # Triton only supports tl.arange when end % 2 == 0
    kv_mask = kv_row_idx < Bc

    q_tile_addrs = tl.arange(0, Br)[:, None] * D + tl.arange(0, D)[None, :]
    kv_tile_addrs = kv_row_idx[:, None] * D + tl.arange(0, D)[None, :]
    o_tile_addrs = tl.arange(0, Br)[:, None] * D + tl.arange(0, D)[None, :]

    for j in tl.range(0, Tc):
        col_idx = j * Bc + kv_row_idx
        seq_mask = col_idx < S
        full_kv_mask = kv_mask & seq_mask

        k = tl.load(k_ptr + j * k_stride + kv_tile_addrs, mask=full_kv_mask[:, None], other=0.0)
        v = tl.load(v_ptr + j * v_stride + kv_tile_addrs, mask=full_kv_mask[:, None], other=0.0)

        for i in tl.range(0, Tr):
            br_offset = i * Br + tl.arange(0, Br)
            mold = tl.load(max_ptr + br_offset)  # [Br, ]
            dnold = tl.load(dn_ptr + br_offset)  # [Br, ]
            oold = tl.load(o_ptr + i * o_stride + o_tile_addrs)  # [Br, D]

            q = tl.load(q_ptr + i * q_stride + q_tile_addrs)  # [Br, D]

            scores = tl.dot(q, tl.trans(k)) * sf  # [Br, D] @ [D, Bc] -> [Br, Bc]
            scores = tl.where(full_kv_mask[None, :], scores, float("-inf"))
            scores = scores.to(tl.float32)

            mcur = tl.max(scores, axis=1)  # [Br, ]
            p = tl.exp(scores - mcur[:, None])  # [Br, Bc]
            dncur = tl.sum(p, axis=1)  # [Br, ]

            mnew = tl.maximum(mcur, mold)

            alpha = tl.exp(mold - mnew)
            beta = tl.exp(mcur - mnew)

            dnnew = dnold * alpha + dncur * beta

            lhs = alpha[:, None] * dnold[:, None] * oold
            rhs = beta[:, None] * tl.dot(p.to(v.dtype), v)

            onew = (lhs + rhs) / dnnew[:, None]

            tl.store(max_ptr + br_offset, mnew)
            tl.store(dn_ptr + br_offset, dnnew)
            tl.store(o_ptr + i * o_stride + o_tile_addrs, onew)


@triton.jit
def _fa_bwd():
    pid = tl.program_id(0)
    pass


class MarineFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim} v.ndim={v.ndim}")

        (B, NH, S, D) = q.shape

        M = torch.cuda.get_device_properties(DEVICE).shared_memory_per_multiprocessor / q.element_size()
        # print(f"M: {M}, Largest square matrix we can fit in shmem: {math.sqrt(M)}")

        Bc = math.ceil(M / (4 * D))
        Bcp = triton.next_power_of_2(Bc)
        Br = min(Bc, D)

        Tr = math.ceil(S / Br)  # number of blocks we divide Q into
        Tc = math.ceil(S / Bc)  # number of blocks we divide K,V into

        # print(f"Br: {Br} Bc: {Bc} Tr: {Tr} Tc: {Tc}")

        m = torch.full((B, NH, S), float("-inf"), dtype=torch.float32, device=q.device)
        dn = torch.zeros((B, NH, S), dtype=torch.float32, device=q.device)
        o = torch.zeros(q.shape, dtype=q.dtype, device=q.device)

        sf = 1 / math.sqrt(D)

        _fa1_fwd[(B, NH)](q, k, v, o, dn, m, sf, B, NH, S, D, Br, Bc, Bcp, Tr, Tc)
        ctx.save_for_backward(o)
        return o

    @staticmethod
    def backward(ctx, dLdc):
        return
