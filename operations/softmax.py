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

    def get_benchmark(self, mode: Any, dtype: Any, **kwargs) -> triton.testing.Benchmark:
        import triton.testing

        m = 4096
        xc = [(2**i) for i in range(8, 15)]
        xc.insert(-1, 12288)

        y_label = "GB/s"

        return triton.testing.Benchmark(
            x_names=["N"],
            x_vals=xc,
            line_arg="kernel",
            line_vals=self.kernels.line_vals(),
            line_names=self.kernels.line_names(),
            styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
            ylabel=y_label,
            plot_name=f"{self.name}-{mode.value}-{dtype.value}",
            args={"M": m, "mode": mode, "torch_dtype": dtype.to_torch()},
        )

    def dims_to_input_args(self, dims: dict, torch_dtype: Any) -> Tuple[Any, ...]:
        return (dims["M"], dims["N"], torch_dtype)
