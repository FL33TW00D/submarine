"""
Simple custom layernorm to learn :)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from marine_ops.marine_ln import MarineLayerNorm

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
    y_tri = MarineLayerNorm.apply(x, w_shape, weight, bias, eps)
    f = torch.compile(
        lambda: F.layer_norm(x, w_shape, weight=weight, bias=bias),
        mode="max-autotune-no-cudagraphs",
    )
    y_ref = f()
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    print("DB Triton: ", db_tri)
    print("DB Ground: ", db_ref)

    print("DW Ground: ", dw_ref)
    print("DW Triton: ", dw_tri)

    print("DX Ground: ", dx_ref)
    print("DX Triton: ", dx_tri)

    # compare
    assert torch.allclose(db_tri, db_ref, atol=5e-4, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-3, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=8e-2, rtol=0)
    assert torch.allclose(y_tri, y_ref, atol=8e-2, rtol=0)


test_layer_norm(2048, 2048, torch.bfloat16)
