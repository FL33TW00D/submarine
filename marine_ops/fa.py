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
def _fa_fwd():
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
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
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor)
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim} v.ndim={v.ndim}")

        (B, NH, S, D) = q.shape

        O = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        _fa_fwd[(B, NH)]()
        ctx.save_for_backward(O)
        return O

    @staticmethod
    def backward(ctx, dLdc):
        return
