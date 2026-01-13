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
def _fa_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    Br: tl.constexpr,
    Bc: tl.constexpr,
    Tr: tl.constexpr,
    Tc: tl.constexpr,
):
    # Each program is responsible for solving one head
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # TODO: Dq and Dv
    b_offset = NH * S * D  # batch offset
    h_offset = S * D  # head offset

    global_offset = pid_b * b_offset + pid_h * h_offset

    q_ptr += global_offset
    k_ptr += global_offset
    v_ptr += global_offset

    l = tl.zeros((S,), dtype=tl.float32)
    m = tl.full((S,), float("-inf"), dtype=tl.float32)

    q_stride = Br * D
    k_stride = Bc * D
    v_stride = Bc * D

    kv_tile_addrs = tl.arange(0, Bc)[:, None] * tl.arange(0, D)[None, :]

    for i in tl.range(0, Tc):
        # load k
        # load v
        # need to be 2d loads
        k = tl.load(k_ptr, i * k_stride + kv_tile_addrs)
        v = tl.load(v_ptr, i * v_stride + kv_tile_addrs)

        for j in tl.range(0, Tr):
            pass

    pass


@triton.jit
def _fa_bwd():
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


class MarineFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim} v.ndim={v.ndim}")

        (B, NH, S, D) = q.shape

        M = torch.cuda.get_device_properties(DEVICE).shared_memory_per_multiprocessor / q.element_size()
        print("M: ", M)
        print("Largest square matrix we can fit in shmem: ", math.sqrt(M))

        Bc = math.ceil(M / (4 * D))
        print("Bc", Bc)
        Br = min(Bc, D)
        print("Br", Br)

        Tr = math.ceil(S / Br)  # number of blocks we divide Q into
        Tc = math.ceil(S / Bc)  # number of blocks we divide K,V into
        print("Tr: ", Tr)
        print("Tc: ", Tc)

        o = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        _fa_fwd[(B, NH)](q, k, v, o, B, NH, S, D, Br, Bc, Tr, Tc)
        ctx.save_for_backward(o)
        return o

    @staticmethod
    def backward(ctx, dLdc):
        return
