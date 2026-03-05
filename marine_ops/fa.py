from hopper.benchmark_attn import q
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

"""
                                 Forward Circuit

  Q [Br,D]                          S [Br,Bc]                P [Br,Bc]                O [Br,D]
    ○──────────── matmul ──────────▶ ○ ──────── softmax ────▶ ○ ─────────────────────▶ ○
                    ▲                                         ▲ 
                    │                                         │
  K [Bc,D]          │                                       matmul
    ○──── trans ────┘                                         │
                                                              │
  V [Bc,D]                                                    │
    ○─────────────────────────────────────────────────────────┘

S = Q @ K^T          [Br,D] x [D,Bc] -> [Br,Bc]
P = softmax(S)       [Br,Bc]
O = P @ V            [Br,Bc] x [Bc,D] -> [Br,D]

Parallel over batch and heads, and sequence in FA2
pid_0 is B*Hq, pid_1 is chunk of Q
"""


@triton.jit
def _fa_fwd(
    q_ptr,  # [B,Hq,T,D]
    k_ptr,  # [B,Hk,S,D]
    v_ptr,  # [B,Hk,S,D]
    out_ptr,  # like(Q)
    lse_ptr,  # logsumexp used for bwd
    sf,  # scale factor
    B,
    Hq,
    S,
    T,
    D: tl.constexpr,
    Br: tl.constexpr,  # Size of Q_tile [Br, D]
    Bc: tl.constexpr,  # Size of KV_tile [Bc, D]
    Tr: tl.constexpr,  # Num Q/O tiles
    Tc: tl.constexpr,  # Num K/V tiles
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    qo_bh_offset = pid_0 * T * D
    qo_t_offset = pid_1 * Br * D
    q_base = q_ptr + qo_bh_offset + qo_t_offset

    q_addrs = q_base + tl.arange(0, Br)[:, None] * D + tl.arange(0, D)[None, :]

    q = tl.load(q_addrs) * sf.to(tl.bfloat16)

    kv_bh_offset = pid_0 * S * D

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + kv_bh_offset,
        shape=(D, S),
        strides=(1, D),
        offsets=(0, 0),
        block_shape=(D, Bc),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + kv_bh_offset,
        shape=(S, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(Bc, D),
        order=(1, 0),
    )

    out = tl.zeros((Br, D), tl.float32)
    denom = tl.zeros((Br,), tl.float32)
    gmax = tl.full((Br,), float("-inf"), tl.float32)

    for i in range(0, Tc):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        scores = tl.dot(q, k).to(tl.float32)

        cmax = tl.max(scores, axis=1)
        nmax = tl.maximum(gmax, cmax)

        p = tl.exp(scores - nmax[:, None])
        d_cur = tl.sum(p, axis=1)

        # if gmax == nmax, tl.exp(0) == 1, no effect
        alpha = tl.exp(gmax - nmax)  # nmax > gmax, e.g tl.exp(-0.5) ~= 0.6, scales down prior contributions
        denom = denom * alpha + d_cur

        out = alpha[:, None] * out + tl.dot(p.to(v.dtype), v)
        gmax = nmax

        k_block_ptr = tl.advance(k_block_ptr, (0, Bc))
        v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))

    out /= denom[:, None]
    lse = tl.log(denom) + gmax

    out_base = out_ptr + qo_bh_offset + qo_t_offset

    out_addrs = out_base + tl.arange(0, Br)[:, None] * D + tl.arange(0, D)[None, :]
    tl.store(out_addrs, out.to(tl.bfloat16))

    lse_addrs = (pid_0 * T) + (pid_1 * Br) + tl.arange(0, Br)
    tl.store(lse_ptr + lse_addrs, lse)


"""
FA2 backward
The only tricky part is the computation of dS, as this requires differentiating through softmax.
Tri has done a nice simplification here.

Pseudocode here: https://arxiv.org/pdf/2307.08691#page=7

Each program is wholly responsible for 1 tile of K & V.
"""


@triton.jit
def _fa_bwd(
    q_ptr,  # input q [B,Hq,T,D]
    k_ptr,  # input k [B,Hk,S,D]
    v_ptr,  # input v [B,Hk,S,D]
    out_ptr,  # input out
    alpha_ptr,  # alpha = rowsum(d0 ○ O) [B, Hq, D]
    lse_ptr,  # input logsumexp
    dout_ptr,  # input dout [B,Hq,T,D]
    dq_ptr,  # output dq like(q)
    dk_ptr,  # output dk like(k)
    dv_ptr,  # output dv like(v)
    B,
    Hq,
    S,
    T,
    D: tl.constexpr,
    Br: tl.constexpr,  # Size of Q_tile [Br, D]
    Bc: tl.constexpr,  # Size of KV_tile [Bc, D]
    Tr: tl.constexpr,  # Num Q/O tiles
    Tc: tl.constexpr,  # Num K/V tiles
):
    pid_0 = tl.program_id(0)  # B*Hq programs
    pid_1 = tl.program_id(1)  # Tc programs

    # Load K and V tile
    kv_off_0 = pid_0 * S * D
    kv_off_1 = pid_1 * Bc * D
    kv_addrs = kv_off_0 + kv_off_1 + (tl.arange(0, Bc)[:, None] * D + tl.arange(0, D))

    k = tl.load(k_ptr + kv_addrs)
    v = tl.load(v_ptr + kv_addrs)

    dK = tl.zeros((Bc, D), tl.float32)
    dV = tl.zeros((Bc, D), tl.float32)

    bhq_off_0 = pid_0 * T * D

    # All of these are the same, traversing a [T, D] subtensor in [Br,D] blocks
    qo_kwargs = {"shape": (T, D), "strides": (D, 1), "offsets": (0, 0), "block_shape": (Br, D), "order": (1, 0)}
    q_bptr = tl.make_block_ptr(base=q_ptr + bhq_off_0, **qo_kwargs)
    dq_bptr = tl.make_block_ptr(base=dq_ptr + bhq_off_0, **qo_kwargs)
    out_bptr = tl.make_block_ptr(base=out_ptr + bhq_off_0, **qo_kwargs)
    dout_bptr = tl.make_block_ptr(base=dout_ptr + bhq_off_0, **qo_kwargs)

    for i in range(0, Tr):
        q = tl.load(q_bptr)  # [Br, D]
        dq = tl.load(dq_bptr)  # [Br, D]
        out = tl.load(out_bptr)  # [Br, D]
        dout = tl.load(dout_bptr)  # [Br, D]

        q_bptr = tl.advance(q_bptr, (Br, 0))
        dq_bptr = tl.advance(dq_bptr, (Br, 0))
        out_bptr = tl.advance(out_bptr, (Br, 0))
        dout_bptr = tl.advance(dout_bptr, (Br, 0))

        pass

    pass


class MarineFA(torch.autograd.Function):
    # scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim} v.ndim={v.ndim}")

        B, Hq, T, D = q.shape
        _, Hk, S, _ = k.shape

        out = torch.empty_like(q)
        lse = torch.empty((B, Hq, T), dtype=torch.float32, device=out.device)

        sf = 1.0 / math.sqrt(D)

        Br = 64  # increasing Br means larger Q chunk is solved, reducing repeated loads of K&V
        Bc = 64  # increasing Bc means larger K chunk is solved, increasing SMEM usage
        Tr = math.ceil(T / Br)
        Tc = math.ceil(T / Bc)

        print(f"Tr: {Tr} Tc: {Tc}")

        compiled = _fa_fwd[(B * Hq, Tr)](
            q,
            k,
            v,
            out,
            lse,
            sf,
            B,
            Hq,
            S,
            T,
            tl.constexpr(D),
            tl.constexpr(Br),
            tl.constexpr(Bc),
            Tr,
            Tc,
            num_stages=2,
        )

        print(f"Physical regs/thread: {compiled.n_regs}")
        print(f"Spills:               {compiled.n_spills}")
        print(f"SMEM:                 {compiled.metadata.shared} bytes")

        ctx.save_for_backward(q, k, v, out, lse)

        return out

    @staticmethod
    def backward(ctx, dLdo):
        q, k, v, out, lse = ctx

        B, Hq, T, D = q.shape
        _, Hk, S, _ = k.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        sf = 1.0 / math.sqrt(D)

        Br = 64
        Bc = 64

        Tr = math.ceil(T / Br)
        Tc = math.ceil(T / Bc)

        alpha = torch.sum(dLdo * out, dim=-1)

        compiled = _fa_bwd[(B * Hq, Tc)](
            q,
            k,
            v,
            out,
            alpha,
            lse,
            dLdo,
            dq,
            dk,
            dv,
            B,
            Hq,
            S,
            T,
            tl.constexpr(D),
            tl.constexpr(Br),
            tl.constexpr(Bc),
            Tr,
            Tc,
        )

        print(f"Physical regs/thread: {compiled.n_regs}")
        print(f"Spills:               {compiled.n_spills}")
        print(f"SMEM:                 {compiled.metadata.shared} bytes")

        ctx.save_for_backward(q, k, v, out, lse)

        return dq, dk, dv
