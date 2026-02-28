import torch
import torch.nn.functional as F
import triton
from marine_ops.cross_entropy import MarineCrossEntropy

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_cross_entropy_fwd(B, T, V, dtype, device=DEVICE):
    logits = torch.randn(B, T, V, dtype=dtype, device=device)
    target = torch.randint(0, V, (B, T), device=device)

    loss_tri = MarineCrossEntropy.apply(logits, target)
    print("Ours forward:  ", loss_tri)

    loss_ref = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
    print("Torch forward: ", loss_ref)

    assert torch.allclose(loss_tri, loss_ref, atol=1e-3, rtol=0)
    print("Forward pass OK")


test_cross_entropy_fwd(B=16, T=100, V=16384, dtype=torch.float32)
