from typing import Callable
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
def block_row_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = reduction_buffer.shape[0]
    col_idx = warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[col_idx] = val  # Only lane 0 of each warp writes
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[lane_idx]  # Only first `warps_per_row` lanes read
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def thread_reduction(x: cute.TensorSSA, op: cute.ReductionOp, init_val: cute.Numeric = 0.0) -> cute.Numeric:
    val = x.reduce(op, init_val=init_val, reduction_profile=0)
    return val


@cute.kernel
def _softmax_fwd(x: cute.Tensor, y: cute.Tensor, warps_per_row: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    log2_e = 1.44269504

    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((warps_per_row,))
    sR = smem.allocate_tensor(cutlass.Float32, smem_layout)

    # bidx == [0..M), tidx == [0..N//vpt)
    tile = x[(None, (bidx, tidx))].load()  # .load() cute.Tensor -> TensorSSA
    tile = tile.to(cutlass.Float32)

    # Max reduction, determine scaling factor for numerical
    thread_max = thread_reduction(tile, cute.ReductionOp.MAX, cute.Float32(float("-inf")))  # 8 -> 1
    warp_max = cute.arch.warp_reduction_max(thread_max)  # 32 -> 1
    block_max = block_row_reduce(warp_max, lambda x, y: cute.arch.fmax(x, y), sR, cute.Float32(float("-inf")))  # N -> 1

    x_shift = tile - block_max
    x_exp = cute.exp2(log2_e * x_shift, fastmath=True)

    # Sum reduction, determine denominator
    tr_exp_sum = thread_reduction(x_exp, cute.ReductionOp.ADD)  # 8 -> 1
    warp_exp_sum = cute.arch.warp_reduction_sum(tr_exp_sum)  # 32 -> 1
    denom = block_row_reduce(warp_exp_sum, lambda x, y: x + y, sR, cute.Float32(0.0))  # N -> 1

    # Now divide it through
    ytile = y[(None, (bidx, tidx))]
    out = (x_exp * cute.arch.rcp_approx(denom)).to(cutlass.BFloat16)
    ytile.store(out)


"""
Kernel launcher
"""


@cute.jit
def softmax(x: cute.Tensor, y: cute.Tensor):
    M, N = x.shape
    # vpt = 128 // x.element_type.width  # LDG 128 BITS is largest coalesced load
    vpt = 32
    tpb = N // vpt
    wpb = tpb // cute.arch.WARP_SIZE
    print(f"vpt={vpt} tpb={tpb} wpb={wpb}")

    zX = cute.zipped_divide(x, (1, vpt))
    zY = cute.zipped_divide(y, (1, vpt))

    _softmax_fwd(zX, zY, wpb).launch(
        grid=(M, 1, 1),
        block=(tpb, 1, 1),
    )


if __name__ == "__main__":
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    x_ = from_dlpack(x, assumed_align=16)
    y_ = from_dlpack(y, assumed_align=16)

    softmax(x_, y_)

    ground = F.softmax(x, dim=-1)
    # print(f"Ours: {y}")
    # print(f"Ground: {ground}")
    torch.testing.assert_close(y, ground)
    print("Values match!")
