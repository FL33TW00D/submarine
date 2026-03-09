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


def test_fa_bwd(B, Hq, Hkv, S, T, D, dtype, device=DEVICE):
    Q = torch.randn((B, Hq, T, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, Hkv, S, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, Hkv, S, D), dtype=dtype, device=device, requires_grad=True)

    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)

    y_tri = MarineFA.apply(Q, K, V)
    y_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref)

    dout = torch.randn_like(y_tri)
    y_tri.backward(dout)
    y_ref.backward(dout)

    for name, g_tri, g_ref in [
        ("dQ", Q.grad, Q_ref.grad),
        ("dK", K.grad, K_ref.grad),
        ("dV", V.grad, V_ref.grad),
    ]:
        # print("REF")
        # print(g_ref)
        # print("")
        # print("OURS")
        # print(g_tri)
        avg_diff = torch.mean(torch.abs(g_ref - g_tri))
        max_diff = torch.max(torch.abs(g_ref - g_tri))
        print(f"{name} max diff: {max_diff} mean diff: {avg_diff}")
        assert torch.allclose(g_tri, g_ref, atol=5e-3, rtol=0), f"{name} mismatch!"

    print("Backward test passed!")


def test_fa_flash(B, Hq, Hkv, S, T, D, dtype, device=DEVICE):
    from flash_attn import flash_attn_func

    # flash_attn_func expects (B, S, H, D)
    Q = torch.randn((B, T, Hq, D), dtype=dtype, device=device)
    K = torch.randn((B, S, Hkv, D), dtype=dtype, device=device)
    V = torch.randn((B, S, Hkv, D), dtype=dtype, device=device)

    y = flash_attn_func(Q, K, V)
    return y


def bench_fa(B, Hq, Hkv, S, T, D, dtype, device=DEVICE):
    from flash_attn import flash_attn_func

    # Ours: (B, H, T, D)
    Q = torch.randn((B, Hq, T, D), dtype=dtype, device=device)
    K = torch.randn((B, Hkv, S, D), dtype=dtype, device=device)
    V = torch.randn((B, Hkv, S, D), dtype=dtype, device=device)

    # Flash: (B, S, H, D)
    Q_flash = Q.transpose(1, 2).contiguous()
    K_flash = K.transpose(1, 2).contiguous()
    V_flash = V.transpose(1, 2).contiguous()

    ms_ours = triton.testing.do_bench(lambda: MarineFA.apply(Q, K, V))
    ms_flash = triton.testing.do_bench(lambda: flash_attn_func(Q_flash, K_flash, V_flash))

    print(f"B={B}, Hq={Hq}, Hkv={Hkv}, S={S}, T={T}, D={D}, dtype={dtype}")
    print(f"  Ours:  {ms_ours:.3f} ms")
    print(f"  Flash: {ms_flash:.3f} ms")
    print(f"  Speedup: {ms_flash / ms_ours:.2f}x")


# test_fa(B=4, Hq=16, Hkv=16, S=2048, T=2048, D=128, dtype=torch.bfloat16)
# test_fa_bwd(B=4, Hq=16, Hkv=16, S=2048, T=2048, D=128, dtype=torch.bfloat16)
# test_fa_flash(B=4, Hq=16, Hkv=16, S=2048, T=2048, D=128, dtype=torch.bfloat16)
bench_fa(B=4, Hq=16, Hkv=16, S=2048, T=2048, D=128, dtype=torch.bfloat16)
