import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import torch.nn.functional as F
import cutlass


"""
Plan:
1. LDG128 from HBM -> Registers
2. Thread reduction using `TensorSSA.reduce`
3. Intra Warp reduction using `shfl_sync_bfly` / xor
4. Block reduction (inter warp), each warp needs a location in smem to write to, elect leader (0 lane), reduce shared, final result per row!

"""


@cute.jit
def wt_reduce(
    x: cute.Tensor,
    op: cute.ReductionOp,
) -> cute.Numeric:
    x = x.load()
    val = x.reduce(op, init_val=float("-inf"), reduction_profile=0)
    val = cute.arch.warp_reduction(
        val,
        max,
    )
    return val


@cute.kernel
def _softmax_fwd(x: cute.Tensor, y: cute.Tensor):
    wv = wt_reduce(x, cute.ReductionOp.MAX) # reduce thread and warp

    


"""
Kernel launcher
"""


@cute.jit
def softmax(x: cute.Tensor, y: cute.Tensor):
    M, N = x.shape
    vpt = 8  # val per thread
    xz = cute.zipped_divide(x, (1, vpt))  # each thread does LDG128, so 8 values (16 * 8 = 128)

    threads_per_block = min(256, N // vpt)

    sA_layout = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout(1024))
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float16, sA_layout, 16)

    _softmax_fwd(xz, y).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )


if __name__ == "__main__":
    M, N = 1024, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y = torch.zeros(M, device="cuda", dtype=torch.float16)
    x_ = from_dlpack(x, assumed_align=16)
    y_ = from_dlpack(y, assumed_align=16)

    softmax(x_, y_)

    ground = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y, ground)
