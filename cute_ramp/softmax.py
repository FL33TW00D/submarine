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
4. Block reduction (inter warp), each warp needs a location in smem to write to, reduce shared, final result per row!
"""


@cute.jit
def block_reduce_single_row(val: cute.Numeric, reduction_buffer: cute.Tensor) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = reduction_buffer.shape[0]
    col_idx = warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[col_idx] = val
    cute.arch.barrier()
    block_reduce_val = cute.Float16(-1000.0)
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[lane_idx]
    return cute.arch.warp_reduction_max(block_reduce_val)


@cute.jit
def thread_reduction(
    x: cute.Tensor,
    op: cute.ReductionOp,
) -> cute.Numeric:
    x = x.load()  # cute.Tensor -> cute.TensorSSA
    val = x.reduce(op, init_val=float("-inf"), reduction_profile=0)
    return val


@cute.kernel
def _softmax_fwd(x: cute.Tensor, y: cute.Tensor, warps_per_row: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((warps_per_row,))
    sR = smem.allocate_tensor(cutlass.Float16, smem_layout)

    # bidx == [0..M), tidx == [0..N//vpt)
    my_tile = x[(None, (bidx, tidx))]
    tr = thread_reduction(my_tile, cute.ReductionOp.MAX)
    vr = cute.arch.warp_reduction_max(tr)
    bmax = block_reduce_single_row(vr, sR)
    if tidx == 0:
        y[bidx] = bmax


"""
Kernel launcher
"""


@cute.jit
def softmax(x: cute.Tensor, y: cute.Tensor):
    M, N = x.shape
    vpt = 128 // x.element_type.width  # LDG 128 BITS is largest coalesced load
    tpb = N // vpt
    wpb = tpb // cute.arch.WARP_SIZE
    print(f"vpt={vpt} tpb={tpb} wpb={wpb}")

    zX = cute.zipped_divide(x, (1, vpt))

    _softmax_fwd(zX, y, wpb).launch(
        grid=(M, 1, 1),
        block=(tpb, 1, 1),
    )


if __name__ == "__main__":
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y = torch.zeros(M, device="cuda", dtype=torch.float16)
    x_ = from_dlpack(x, assumed_align=16)
    y_ = from_dlpack(y, assumed_align=16)

    softmax(x_, y_)

    ground = torch.max(x, dim=1)[0]
    # print(f"Ours: {y}")
    # print(f"Ground: {ground}")
    torch.testing.assert_close(y, ground)
    # print("Values match!")
