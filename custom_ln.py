"""
Simple custom layernorm to learn :)
"""

import torch
import triton
import triton.language as tl

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=["N", "BLOCK_SIZE"],
)
@triton.jit
def _layer_norm_fwd_fused(
    x_ptr,  # pointer to the input
    y_ptr,  # pointer to the output
    w_ptr,  # pointer to the weights
    b_ptr,  # pointer to the biases
    mu_ptr,  # pointer to the mean
    rstd_ptr,  # pointer to the 1/std
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    local_mask = offs < N
    irange = pid * N + offs
    x = tl.load(x_ptr + irange, local_mask, other=0.0).to(tl.float32)
    mu = tl.sum(x) / N
    tl.store(mu_ptr + pid, mu)

    x_shift = tl.where(local_mask, x - mu, 0.0)

    var = tl.sum(x_shift * x_shift) / N

    rstd = tl.rsqrt(var + eps)

    tl.store(rstd_ptr + pid, rstd)

    w = tl.load(w_ptr + offs, local_mask).to(tl.float32)
    b = tl.load(b_ptr + offs, local_mask).to(tl.float32)

    x_norm = tl.fma((x_shift * rstd), w, b)
    tl.store(y_ptr + irange, x_norm.to(OUT_DT), local_mask)


@triton.jit
def _layer_norm_bwd_fused(
    dLdy_ptr,  # pointer to dLdy
    x_ptr,  # pointer to input
    w_ptr,  # pointer to weight
    mu_ptr,  # pointer to cached mu
    rstd_ptr,  # pointer to cached rstd
    dx_ptr,  # pointer to output dX
    dw_ptr,  # pointer to output dW
    db_ptr,  # pointer to output dB
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    # load x, w, mean, rstd
    pid = tl.program_id(0)
    mu = tl.load(mu_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    print(rstd)


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE = calculate_settings(N)
        _layer_norm_fwd_fused[(M,)](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mu, rstd = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        dw = torch.empty((N,), dtype=x.dtype, device=x.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)
        BLOCK_SIZE = calculate_settings(N)

        _layer_norm_bwd_fused[(M,)](
            dLdy,
            x,
            w,
            mu,
            rstd,
            dLdx,
            dw,
            db,
            N,
            BLOCK_SIZE,
            OUT_DT=tch_to_trt[x.dtype],
        )


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
