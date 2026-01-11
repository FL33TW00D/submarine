from marine_ops.gemm import MarineGEMM
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


def test_gemm(M, N, dtype, device=DEVICE):
    K = 1024

    A = torch.randn((M, K), dtype=dtype, device=DEVICE)
    B = torch.randn((K, N), dtype=dtype, device=DEVICE)

    y_tri = MarineGEMM.apply(A, B)
    print("Ours forward: ", y_tri)
    f = torch.compile(
        lambda: torch.matmul(A, B),
        mode="max-autotune-no-cudagraphs",
    )
    y_ref = f()
    print("Torch forward: ", y_ref)
    assert torch.allclose(y_tri, y_ref, atol=1e-3, rtol=0)

    assert torch.allclose(y_tri, y_ref, atol=1e-4, rtol=0)


test_gemm(2048, 2048, torch.bfloat16)
