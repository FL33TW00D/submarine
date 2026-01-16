import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import torch.nn.functional as F
from typing import Callable
import math
import cutlass


@cute.jit
def row_reduce(
    x: cute.Tensor,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
) -> cute.Numeric:
    x = x.load()
    val = x.reduce(op, init_val=float("-inf"), reduction_profile=0)
    val = cute.arch.warp_reduction(
        val,
        max,
        threads_in_group=min(32, cute.arch.WARP_SIZE),
    )
    return val


if __name__ == "__main__":
    M, N = 1024, 1024
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, device="cuda", dtype=torch.float16)
    a_ = from_dlpack(a, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    x = row_reduce(a_, cute.ReductionOp.MAX, 256)
    print(x)
    ground = torch.max(a, dim=1)[0]
    torch.testing.assert_close(c, ground)
