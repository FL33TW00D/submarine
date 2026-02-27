from marine_ops.rope import MarineRoPE
from marine_ops.fa import MarineFA
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class RoPE(nn.Module):
    dim: int
    base: int

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.base = 10000
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim))
        t = torch.arange(0, max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # shape: (max_seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("sin_cached", torch.sin(emb).to("cuda"), persistent=False)
        self.register_buffer("cos_cached", torch.cos(emb).to("cuda"), persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = q.shape
        assert D == self.dim

        sin = self.sin_cached
        cos = self.cos_cached
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


def test_rope(B, S, D, dtype, device=DEVICE):
    """
    Test Rope implementation.

    Args:
        B: Batch size
        S: Sequence length
        D: Head dimension
        dtype: Data type
        device: Device to run on
    """
    q = torch.randn((B, S, D), dtype=dtype, device=device)
    k = torch.randn((B, S, D), dtype=dtype, device=device)

    ref = RoPE(D, S)
    (rq_rot, rk_rot) = ref(q, k)
    rq_rot = rq_rot.to(dtype)
    rk_rot = rk_rot.to(dtype)

    (oq_rot, ok_rot) = MarineRoPE.apply(q, k, ref.sin_cached, ref.cos_cached)

    print("THEIRS: ", rq_rot)
    print("OURS: ", oq_rot)
    assert torch.allclose(rq_rot, oq_rot, atol=1e-3, rtol=0)
    assert torch.allclose(rk_rot, ok_rot, atol=1e-3, rtol=0)
    print("Test passed!")


test_rope(B=4, S=512, D=64, dtype=torch.bfloat16)
