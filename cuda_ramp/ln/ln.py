import time
import torch
from torch.utils.cpp_extension import include_paths

header_code = open("ln.cuh").read()
kernel_source = open("ln.cu").read()

t0 = time.perf_counter()
kernel = torch.cuda._compile_kernel(
    kernel_source,
    "ln_kernel",
    header_code=header_code,
    cuda_include_dirs=include_paths("cuda"),
)
print(f"NVRTC compile: {(time.perf_counter() - t0) * 1000:.1f}ms")

T, D = 64, 2048
TPB = 256
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)

num_warps = TPB // 32
kernel(
    (1, T, 1),
    (TPB, 1, 1),
    (x, out, 1e-5, D),
    shared_mem=num_warps * 4,
)

torch.testing.assert_close(out, torch.layer_norm(x, (D,)), atol=1e-2, rtol=1e-2)
print("passed")
