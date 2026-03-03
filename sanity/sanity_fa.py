from marine_ops.fa import MarineFA
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def test_fa(B, Hq, Hkv, S, T, D, dtype, device=DEVICE):
    Q = torch.randn((B, Hq, T, D), dtype=dtype, device=device)
    K = torch.randn((B, Hkv, S, D), dtype=dtype, device=device)
    V = torch.randn((B, Hkv, S, D), dtype=dtype, device=device)

    y_tri = MarineFA.apply(Q, K, V)

    # y_ref = F.scaled_dot_product_attention(Q, K, V)
    # print("Ours forward: ", y_tri)
    # print("Torch forward: ", y_ref)

    # max_diff = torch.max(torch.abs(y_ref - y_tri))
    # print(max_diff)

    # assert torch.allclose(y_tri, y_ref, atol=5e-3, rtol=0)
    # print("Test passed!")


test_fa(B=4, Hq=8, Hkv=8, S=512, T=512, D=64, dtype=torch.bfloat16)
