"""Benchmark: CustomLayerNorm vs torch native vs liger."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerLayerNorm

from triton.testing import do_bench

# from custom_ln import CustomLayerNorm
from custom_ln import CustomLayerNormContig as CustomLayerNorm

TOL = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 7e-2}


def check(name, y, y_ref, dtype):
    tol = TOL[dtype]
    if not torch.allclose(y, y_ref, atol=tol, rtol=tol):
        diff = (y - y_ref).abs().max().item()
        raise AssertionError(f"{name} failed: max_diff={diff:.2e}, tol={tol:.2e}")


def bench(m, n, dtype=torch.float16):
    # x = torch.randn(m, n, device="cuda", dtype=dtype)
    x = torch.randn(m, n * 2, device="cuda", dtype=dtype)[:, ::2]  # non-contiguous
    w = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)

    liger_ln = LigerLayerNorm(n).cuda()
    liger_ln.weight.data = w
    liger_ln.bias.data = b

    # correctness
    y_torch = F.layer_norm(x, (n,), w, b)
    y_custom = CustomLayerNorm.apply(x, (n,), w, b)
    y_liger = liger_ln(x)
    check("custom", y_custom, y_torch, dtype)
    check("liger", y_liger, y_torch, dtype)

    # speed
    ms_torch = do_bench(lambda: F.layer_norm(x, (n,), w, b))
    ms_custom = do_bench(lambda: CustomLayerNorm.apply(x, (n,), w, b))
    ms_liger = do_bench(lambda: liger_ln(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms / 1e6
    return gbps(ms_torch), gbps(ms_custom), gbps(ms_liger)


sizes = [
    (128, 1024),
    (256, 2048),
    (512, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (4096, 8192),
    (8192, 8192),
    (16384, 8192),
]

labels = [f"{m}x{n}" for m, n in sizes]
results = []
for m, n in sizes:
    print(f"{m}x{n}: ", end="", flush=True)
    results.append(bench(m, n))
    print("ok")
torch_gbps, custom_gbps, liger_gbps = zip(*results)

plt.figure(figsize=(12, 5))
plt.plot(labels, torch_gbps, "o-", label="torch", linewidth=2, markersize=6)
plt.plot(labels, custom_gbps, "s-", label="custom", linewidth=2, markersize=6)
plt.plot(labels, liger_gbps, "^-", label="liger", linewidth=2, markersize=6)
plt.xlabel("Shape (M x N)")
plt.ylabel("GB/s")
plt.title("LayerNorm Throughput Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("bench_ln.png")
print("Saved bench_ln.png")
