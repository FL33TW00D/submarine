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


@triton.autotune(
    configs=[
        triton.Config(kwargs={"Br": 64, "Bc": 64}),
        triton.Config(kwargs={"Br": 32, "Bc": 64}),
        triton.Config(kwargs={"Br": 128, "Bc": 64}),
        triton.Config(kwargs={"Br": 64, "Bc": 128}),
        triton.Config(kwargs={"Br": 64, "Bc": 32}),
        triton.Config(kwargs={"Br": 256, "Bc": 32}),
        triton.Config({"Br": 128, "Bc": 64}, num_warps=4, num_stages=3),
        triton.Config({"Br": 128, "Bc": 64}, num_warps=8, num_stages=3),
        triton.Config({"Br": 128, "Bc": 64}, num_warps=8, num_stages=3),
        triton.Config({"Br": 64, "Bc": 64}, num_warps=4, num_stages=4),
        triton.Config({"Br": 64, "Bc": 64}, num_warps=8, num_stages=2),
    ],
    key=["T", "S"],  # the two above configs will be evaluated anytime
    # the value of x_size changes
)
@triton.jit
def _fa_fwd(
    q_ptr,  # [B, Hq, T, D]
    k_ptr,  # [B, Hk, S, D]
    v_ptr,
    out_ptr,
    lse_ptr,  # [B, Hq, T]
    sf,
    B: tl.constexpr,
    Hq: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    Hk: tl.constexpr,
    S: tl.constexpr,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    Tr = tl.cdiv(T, Br)
    Tc = tl.cdiv(S, Bc)
    # Each program solves a chunk of Q sequence
    # Requires loading the FULL K and V sequence, so increasing Br is good for memory reuse
    # Must be on hopper to get benefit from GQA, as head mats can be multicast to multiple SMs
    log2_e = 1.4426950409  # exp(x) = 2^log2(e)*x

    pid_0 = tl.program_id(0)  # B*Hq
    pid_1 = tl.program_id(1)  # Tr

    bh_offset = pid_0 * T * D
    br_offset = pid_1 * Br * D

    qo_addrs = bh_offset + br_offset + tl.arange(0, Br)[:, None] * D + tl.arange(0, D)

    qk_scale = sf
    qk_scale *= log2_e

    q = (tl.load(q_ptr + qo_addrs).to(tl.float32) * qk_scale).to(tl.bfloat16)  # TODO: mask

    out = tl.zeros((Br, D), tl.float32)
    gmax = tl.full((Br,), float("-inf"), tl.float32)
    lse = tl.zeros((Br,), tl.float32)

    for t in tl.range(0, Tc, warp_specialize=True):
        kv_addrs = bh_offset + (t * Bc * D) + tl.arange(0, Bc)[:, None] * D + tl.arange(0, D)
        k = tl.load(k_ptr + kv_addrs)  # TODO: mask

        s = tl.dot(q, tl.trans(k))  # [Br, Bc]

        cmax = tl.max(s, axis=-1)
        nmax = tl.maximum(gmax, cmax)  # [Br]

        p = tl.exp2(s - nmax[:, None])
        alpha = tl.exp2(gmax - nmax)  # nmax > gmax, e.g exp(-0.5) ~= 0.6, scales down prior contributions
        lse = lse * alpha + tl.sum(p, axis=-1)
        gmax = nmax

        v = tl.load(v_ptr + kv_addrs)
        out = out * alpha[:, None] + tl.dot(p.to(v.dtype), v)

    out /= lse[:, None]
    lse = gmax + tl.log(lse)

    tl.store(out_ptr + qo_addrs, out.to(tl.bfloat16))

    lse_addrs = (pid_0 * T) + (pid_1 * Br) + tl.arange(0, Br)
    tl.store(lse_ptr + lse_addrs, lse)


"""
FA2 backward
The only tricky part is the computation of dS, as this requires differentiating through softmax.
Tri has done a nice simplification here.

Pseudocode here: https://arxiv.org/pdf/2307.08691#page=7

Each program is wholly responsible for 1 tile of K & V [Bc, D]
"""


@triton.jit
def _fa_bwd():
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

        # Parallelize over batch and heads and sequence in FA2
        # Using Tr means parallelizing over query
        grid = lambda meta: (B * Hq, triton.cdiv(T, meta["Br"]))

        compiled = _fa_fwd[grid](
            q,
            k,
            v,
            out,
            lse,
            sf,
            B,
            Hq,
            T,
            D,
            Hk,
            S,
        )
        print(_fa_fwd.best_config)

        # print(f"Physical regs/thread: {compiled.n_regs}")
        # print(f"Spills:               {compiled.n_spills}")
        # print(f"SMEM:                 {compiled.metadata.shared} bytes")

        ctx.save_for_backward(q, k, v, out, lse)

        return out

    @staticmethod
    def backward(ctx, dLdo):
        q, k, v, out, lse = ctx.saved_tensors

        B, Hq, T, D = q.shape
        _, Hk, S, _ = k.shape

        compiled = _fa_bwd[()]()

        print(f"Physical regs/thread: {compiled.n_regs}")
        print(f"Spills:               {compiled.n_spills}")
        print(f"SMEM:                 {compiled.metadata.shared} bytes")

        ctx.save_for_backward(q, k, v, out, lse)

        return
