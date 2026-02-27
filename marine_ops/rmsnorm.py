from typing import Optional, Self

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from hypothesis import given, settings, strategies as st

DEVICE = triton.runtime.driver.active.get_active_torch_device()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _rms_norm_fwd_fused(
    x_ptr,  # pointer to the input [M,N]
    w_ptr,  # pointer to weight [N,]
    rrms_ptr,  # pointer to 1/RMS [M,]
    y_ptr,  # pointer to the output [M,N]
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    # Load X
    # Load M
    pid = tl.program_id(0)
    offset = pid * N

    rnge = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offset + rnge, rnge < N, 0.0).to(tl.float32)
    w = tl.load(w_ptr + rnge, rnge < N).to(tl.float32)

    mu_sq = tl.sum((x * x).cast(tl.float32)) * (1 / N)
    rrms = tl.rsqrt(mu_sq + eps)

    tl.store(rrms_ptr + pid, rrms)

    y = rrms * x * w
    tl.store(y_ptr + offset + rnge, y.to(OUT_DT), rnge < N)


"""
1/RMS(v . ((v . xhat) / N). xhat)
"""


@triton.jit
def _rms_norm_bwd_fused(
    dLdy_ptr,  # pointer to dLdy
    x_ptr,  # pointer to input [M, N]
    w_ptr,  # pointer to weight [N,]
    N,  # number of columns in X
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


class MarineRMSNorm(torch.autograd.Function):
    TOLS = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 7e-2}

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rrms = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = calculate_settings(N)
        _rms_norm_fwd_fused[(M,)](  #
            x_arg,
            weight,
            rrms,
            y,
            N,
            eps,
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            OUT_DT=tl.constexpr(tch_to_trt[x.dtype]),
        )
        ctx.save_for_backward(x, rrms)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        return (None,) * 5

    """
    Property testing for validating forward across a wide range of shapes
    """

    @classmethod
    def validate_fwd(cls: type[Self], max_examples: int = 100):
        @given(
            M=st.integers(min_value=8, max_value=4096),
            N=st.integers(min_value=8, max_value=4096),
            dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
            data_seed=st.integers(min_value=0, max_value=2**32 - 1),
        )
        @settings(max_examples=max_examples, deadline=None, database=None)
        def _validate(M: int, N: int, dtype: torch.dtype, data_seed: int):
            torch.manual_seed(data_seed)

            x = torch.rand((M, N), device=DEVICE, dtype=dtype)
            norm_shape = (x.shape[-1],)
            weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
            bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)

            y_custom = MarineRMSNorm.apply(x, norm_shape, weight, bias)
            y_torch = F.rms_norm(x, norm_shape, weight, bias)

            assert torch.allclose(y_custom, y_torch, atol=cls.TOLS[dtype], rtol=cls.TOLS[dtype]), (
                f"Mismatch for shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(y_custom - y_torch).abs().max().item():.2e}"
            )

        _validate()

    @classmethod
    def validate_bwd(cls: type[Self], max_examples: int = 100):
        @given(
            M=st.integers(min_value=8, max_value=4096),
            N=st.integers(min_value=8, max_value=4096),
            dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
            data_seed=st.integers(min_value=0, max_value=2**32 - 1),
        )
        @settings(max_examples=max_examples, deadline=None, database=None)
        def _validate(M: int, N: int, dtype: torch.dtype, data_seed: int):
            torch.manual_seed(data_seed)

            x = torch.rand((M, N), device=DEVICE, dtype=dtype)
            norm_shape = (x.shape[-1],)
            weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
            bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
            dy = 0.1 * torch.randn_like(x)

            x.requires_grad_(True)
            weight.requires_grad_(True)
            bias.requires_grad_(True)

            y_custom = MarineRMSNorm.apply(x, norm_shape, weight, bias)
            y_custom.backward(dy, retain_graph=True)
            dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]

            y_torch = F.rms_norm(x, norm_shape, weight)
            y_torch.backward(dy, retain_graph=True)
            dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]

            assert torch.allclose(dx_ref, dx_tri, atol=cls.TOLS[dtype], rtol=cls.TOLS[dtype]), (
                f"Mismatch for DX, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
            )
            assert torch.allclose(dw_ref, dw_tri, atol=cls.TOLS[dtype], rtol=cls.TOLS[dtype]), (
                f"Mismatch for DW, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
            )
            assert torch.allclose(db_ref, db_tri, atol=cls.TOLS[dtype], rtol=cls.TOLS[dtype]), (
                f"Mismatch for DB, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
            )

        _validate()
