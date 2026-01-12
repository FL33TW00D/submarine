from typing import Tuple
import torch
import triton
import triton.language as tl
import math

DEVICE = triton.runtime.driver.active.get_active_torch_device()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _gemm_fwd(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, 8)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        out += tl.dot(a, b)

    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


@triton.jit
def _gemm_bwd():
    pid = tl.program_id(0)
    pass


def calculate_block_size(M: int, N: int) -> Tuple[int, int, int]:
    # Wave quantization is the root of all evil, and this is a function of the number of SMs and our dispatch
    # We need enough warps to be assigned to each SM to latency hide
    # So, query the number of SMs
    # On Ampere we have 4 tensor cores per SM.

    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    # our atom is m16n8k16
    # 500 cycles to GMEM
    # Littles Law, need 32 warps to hide the latency

    # So 32 * sm_count (3090 == 82) = minimum 2624 warps
    # So if we had 32x32 tiles, minimum matrix size is sqrt(2624*(32*32)) ~ 1600x1600
    # 32 warps per SM

    return (64, 256, 32)


class MarineGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor):
        if A.ndim < 2 or B.ndim < 2:
            # TODO: should just reshape
            raise ValueError(f"Expected 2D+ tensors, got A.ndim={A.ndim}, B.ndim={B.ndim}")

        # TODO: handle 2d -> 3d/4d reshape shit
        M = A.shape[-2]
        N = B.shape[-1]
        K = A.shape[-1]
        output_shape = (A.shape[-2], B.shape[-1])
        # TODO: dtype handling
        C = torch.empty(output_shape, dtype=A.dtype, device=A.device)

        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) = calculate_block_size(M, N)

        MD = math.ceil(M / BLOCK_SIZE_M)
        ND = math.ceil(N / BLOCK_SIZE_N)

        _gemm_fwd[(MD, ND)](A, B, C, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        ctx.save_for_backward(A, B, C)
        return C

    @staticmethod
    def backward(ctx, dLdc):
        return
