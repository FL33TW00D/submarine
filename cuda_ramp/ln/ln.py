import time
import torch

src = open("ln.cu").read()

t0 = time.perf_counter()
kernel = torch.cuda._compile_kernel(src, "ln_kernel")
print(f"NVRTC compile: {(time.perf_counter() - t0) * 1000:.1f}ms")

T, D = 64, 2048
TPB = 256

x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)

num_warps = TPB // 32
shared_mem = num_warps * 4  # num_warps * sizeof(float)

kernel((1, T, 1), (TPB, 1, 1), (x, out, 1e-5, D), shared_mem=shared_mem)

torch.testing.assert_close(out, torch.layer_norm(x, (D,)), atol=1e-2, rtol=1e-2)
print("passed")
