import torch
import triton
import triton.language as tl

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _gemm_fwd():
    pid = tl.program_id(0)
    pass


@triton.jit
def _gemm_bwd():
    pid = tl.program_id(0)


def calculate_settings(n: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    return BLOCK_SIZE


class MarineGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError(f"Expected 2D+ tensors, got a.ndim={a.ndim}, b.ndim={b.ndim}")
            # How best to handle this here?
            return
        M = a.shape[-2]
        N = b.shape[-1]
        output_shape = (a.shape[-2], b.shape[-1])
        y = torch.empty(output_shape, dtype=a.dtype, device=a.device)
        # reshape input data into 2D tensor
        BLOCK_SIZE = calculate_settings(N)
        # TODO: correctly compute num programs
        _gemm_fwd[(M,)]()
        ctx.save_for_backward(a, b)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dLdc):
        return
