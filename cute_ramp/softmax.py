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

# Confusions right now:
1. SMEM layout. Each thread reduction goes from 8 -> 1 value, we need to place those values into SMEM. We want 
to avoid bank conflicts. Each bank can hold 4B of data. We should therefore pad it out, such that each thread is
accessing consecutive banks. How to do it?
"""


@cute.jit
def t_reduce(
    x: cute.Tensor,
    op: cute.ReductionOp,
) -> cute.Numeric:
    x = x.load()  # cute.Tensor -> cute.TensorSSA
    val = x.reduce(op, init_val=float("-inf"), reduction_profile=0)
    return val


@cute.kernel
def _softmax_fwd(x: cute.Tensor, y: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = x.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n

    tr = t_reduce(x[(None, (mi, ni))], cute.ReductionOp.MAX)  # reduce thread and warp
    tv_layout = cute.make_layout(shape=((2, 2), (2, 3)), stride=((2, 12), (1, 4)))
    composed = cute.composition(tensor, tv_layout)
    my_data = composed[(tidx, None)]  # Just use local tidx!


"""
Kernel launcher
"""


@cute.jit
def softmax(x: cute.Tensor, y: cute.Tensor):
    M, N = x.shape
    vpt = 8  # val per thread
    xz = cute.zipped_divide(x, (1, vpt))  # each thread does LDG128, so 8 values (16 * 8 = 128)
    print(xz)

    threads_per_block = min(256, N // vpt)
    num_warps = threads_per_block // cute.arch.WARP_SIZE

    smem = cutlass.utils.SmemAllocator()
    # Each thread does an LDG128 and loads 8 values.
    # It then does an 8 value reduction
    # Then we do a warp reduction, so 32 threads get their values reduced to 1
    # They then write this into shared memory

    smem_layout = cute.make_layout((1, num_warps))
    sR = smem.allocate_tensor(cutlass.Float16, smem_layout)

    print(f"Threads per block {threads_per_block}")
    print(f"Num warps: {num_warps}")

    _softmax_fwd(xz, y).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )


if __name__ == "__main__":
    M, N = 2048, 2048
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y = torch.zeros(M, device="cuda", dtype=torch.float16)
    x_ = from_dlpack(x, assumed_align=16)
    y_ = from_dlpack(y, assumed_align=16)

    softmax(x_, y_)

    ground = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y, ground)
