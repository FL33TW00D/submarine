import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from marine_ops.marine_softmax import MarineSoftmax

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_softmax(M, N, dtype, device=DEVICE):
    x_shape = (M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)

    x.requires_grad_(True)
    # forward pass
    y_tri = MarineSoftmax.apply(x)
    print("Ours forward: ", y_tri)
    f = torch.compile(
        lambda: F.softmax(x, dim=-1),
        mode="max-autotune-no-cudagraphs",
    )
    y_ref = f()
    print("Torch forward: ", y_ref)
    assert torch.allclose(y_tri, y_ref, atol=1e-3, rtol=0)

    y_tri.backward(dy, retain_graph=True)
    dx_tri = x.grad.clone()
    x.grad = None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref = x.grad.clone()

    print("DX Ground: ", dx_ref)
    print("DX Triton: ", dx_tri)

    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-4, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-4, rtol=0)


test_softmax(2048, 2048, torch.bfloat16)
