import torch

kernel_src = open("softmax.cu").read()

kernel = torch.cuda._compile_kernel(kernel_src, "softmax_kernel")

M, N = 64, 128
x = torch.randn(M, N, device="cuda")
out = torch.empty_like(x)

TPB = 32
kernel((M, 1, 1), (TPB, 1, 1), (x, out, N))

torch.testing.assert_close(out, torch.softmax(x, dim=-1))
print("passed")
