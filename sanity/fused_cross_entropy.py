import torch
import torch.nn.functional as F
import triton
from marine_ops.fused_cross_entropy import MarineLinearCrossEntropy

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_fused_cross_entropy_fwd(BT, D, V, dtype, device=DEVICE):
    # x is (BT, D) — hidden states going into the final linear layer
    x = torch.randn(BT, D, dtype=dtype, device=device)
    weight = torch.randn(V, D, dtype=dtype, device=device)
    target = torch.randint(0, V, (BT,), device=device)

    loss_tri = MarineLinearCrossEntropy.apply(x, weight, target)
    print("Ours forward:  ", loss_tri)

    # Reference: fused linear + cross-entropy
    logits = x @ weight.T  # (BT, V)
    loss_ref = F.cross_entropy(logits.to(torch.float32), target, reduction="none").to(torch.float32)
    print("Torch forward: ", loss_ref)

    assert torch.allclose(loss_tri, loss_ref, atol=5e-1, rtol=1e-3), (
        f"Max diff: {(loss_tri - loss_ref).abs().max().item()}"
    )
    print("Forward pass OK")


test_fused_cross_entropy_fwd(BT=16384, D=512, V=4096, dtype=torch.bfloat16)
