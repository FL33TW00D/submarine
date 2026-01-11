from operations.operation import Operation, KernelEnum
from typing import Callable, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from marine_ops.gemm import MarineGEMM
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class GEMMKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    CUSTOM = "custom"


class GEMMOp(Operation):
    @property
    def name(self) -> str:
        return "gemm"

    @property
    def kernels(self) -> type[GEMMKernels]:
        return GEMMKernels

    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (a, b, dLdc) = inputs
        match kernel:
            case GEMMKernels.TCH:
                ref = torch.matmul(a, b)
                f = lambda: ref.backward(dLdc, retain_graph=True)
            case GEMMKernels.CUSTOM:
                ref = MarineGEMM.apply(a, b)
                f = lambda: ref.backward(dLdc, retain_graph=True)
            case GEMMKernels.TCH_CMP:
                ref = torch.matmul(a, b)
                f = torch.compile(
                    lambda: ref.backward(dLdc, retain_graph=True),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(3):
                    f()  # warm up
        return f

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (a, b) = inputs

        match kernel:
            case GEMMKernels.TCH:
                f = lambda: torch.matmul(a, b)
            case GEMMKernels.CUSTOM:
                f = lambda: MarineGEMM.apply(a, b)
            case GEMMKernels.TCH_CMP:
                f = torch.compile(
                    lambda: torch.matmul(a, b),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(5):
                    f()  # warm up

        return f

    def generate_fwd_inputs(self, args: Any):
        (M, N, dtype) = args
        torch.manual_seed(0)
        A = torch.randn((M, 1024), dtype=dtype, device=DEVICE)
        B = torch.randn((1024, N), dtype=dtype, device=DEVICE)
        return (A, B)

    def generate_bwd_inputs(self, args: Any):
        (M, N, K, dtype) = args
        torch.manual_seed(0)
        A = torch.randn((M, K), dtype=dtype, device=DEVICE)
        B = torch.randn((K, N), dtype=dtype, device=DEVICE)
        A.requires_grad_(True)
        B.requires_grad_(True)
        dLdc = 0.1 * torch.randn((M, K), dtype=dtype, device=DEVICE)
        return (A, B, dLdc)
