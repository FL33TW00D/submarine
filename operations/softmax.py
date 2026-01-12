from operations.operation import Operation, KernelEnum
from typing import Callable, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerSoftmax
from marine_ops.softmax import MarineSoftmax
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class SoftmaxKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    LIGER = "liger"
    CUSTOM = "custom"


class SoftmaxOp(Operation):
    @property
    def name(self) -> str:
        return "softmax"

    @property
    def memory_bound(self) -> bool:
        return True

    @property
    def kernels(self) -> type[SoftmaxKernels]:
        return SoftmaxKernels

    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (x, dLdy) = inputs
        match kernel:
            case SoftmaxKernels.TCH:
                ref = F.softmax(x, dim=-1)
                f = lambda: ref.backward(dLdy, retain_graph=True)
            case SoftmaxKernels.CUSTOM:
                ref = MarineSoftmax.apply(x, -1)
                f = lambda: ref.backward(dLdy, retain_graph=True)
            case SoftmaxKernels.TCH_CMP:
                ref = F.softmax(x, dim=-1)
                f = torch.compile(
                    lambda: ref.backward(dLdy, retain_graph=True),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(3):
                    f()  # warm up
            case SoftmaxKernels.LIGER:
                soft = LigerSoftmax()
                ref = soft(x)
                f = lambda: ref.backward(dLdy, retain_graph=True)
        return f

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (x,) = inputs

        match kernel:
            case SoftmaxKernels.TCH:
                f = lambda: F.softmax(x, dim=-1)
            case SoftmaxKernels.CUSTOM:
                f = lambda: MarineSoftmax.apply(x, -1)
            case SoftmaxKernels.TCH_CMP:
                f = torch.compile(
                    lambda: F.softmax(x, dim=-1),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(5):
                    f()  # warm up
            case SoftmaxKernels.LIGER:
                soft = LigerSoftmax()
                f = lambda: soft(x)

        return f

    def generate_fwd_inputs(self, args: Any):
        (M, N, dtype) = args
        torch.manual_seed(0)
        x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=DEVICE)
        return (x,)

    def generate_bwd_inputs(self, args: Any):
        (x,) = self.generate_fwd_inputs(args)
        x.requires_grad_(True)
        dy = 0.1 * torch.randn_like(x)
        return (x, dy)

    def fwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        (x, *_) = inputs
        return lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    def bwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        (x, *_) = inputs
        return lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
