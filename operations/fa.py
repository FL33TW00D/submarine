from marine_ops.fa import MarineFA
from operations.operation import Operation, KernelEnum
from typing import Callable, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class FAKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    CUSTOM = "custom"


class FAOp(Operation):
    @property
    def name(self) -> str:
        return "fa"

    @property
    def memory_bound(self) -> bool:
        # Depends, for MHA it's true
        return True

    @property
    def kernels(self) -> type[FAKernels]:
        return FAKernels

    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        return lambda f: None

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (q, k, v) = inputs

        match kernel:
            case FAKernels.TCH:
                f = lambda: F.scaled_dot_product_attention(q, k, v)
            case FAKernels.CUSTOM:
                f = lambda: MarineFA.apply(q, k, v)
            case FAKernels.TCH_CMP:
                f = torch.compile(
                    lambda: F.scaled_dot_product_attention(q, k, v),
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
        pass

    def fwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        return None

    def bwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        return None
