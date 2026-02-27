import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from marine_ops.rmsnorm import MarineRMSNorm

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_rms_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    norm_shape = (x_shape[-1],)
    weight = torch.rand(norm_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = MarineRMSNorm.apply(x, norm_shape, weight, eps)
    f = torch.compile(
        lambda: F.rms_norm(x, norm_shape, weight=weight, eps=eps),
        mode="max-autotune-no-cudagraphs",
    )
    y_ref = f()
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    (dx_tri,) = [_.grad.clone() for _ in [x]]
    x.grad, weight.grad = None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [_.grad.clone() for _ in [x, weight]]

    print("DX Ground: ", dx_ref)
    print("DX Triton: ", dx_tri)

    # compare
    assert torch.allclose(dx_tri, dx_ref, atol=8e-2, rtol=0)
    print(f"ours: {y_tri}")
    print(f"ground: {y_ref}")
    assert torch.allclose(y_tri, y_ref, atol=5e-2, rtol=0)


test_rms_norm(2048, 2048, torch.bfloat16)
