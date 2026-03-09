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
def _fa_fwd():
    pass


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

        compiled = _fa_fwd[()]()

        print(f"Physical regs/thread: {compiled.n_regs}")
        print(f"Spills:               {compiled.n_spills}")
        print(f"SMEM:                 {compiled.metadata.shared} bytes")

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
