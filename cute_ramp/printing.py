import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import torch.nn.functional as F


@cute.kernel
def print_kernel(t: cute.Tensor):
    print("in kernel", t)
    cute.printf(t)
    cute.print_tensor(t)


@cute.jit
def print_t(t: cute.Tensor):
    print("in jit", t)
    print_kernel(t).launch(grid=(1, 1, 1), block=(1, 1, 1))


a = torch.randn(8, device="cuda")
print_t(cute.runtime.from_dlpack(a))
