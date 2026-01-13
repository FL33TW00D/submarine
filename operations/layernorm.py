from operations.operation import Operation, KernelEnum
from marine_ops.layernorm import MarineLayerNorm
from typing import Callable, Any, Tuple, Optional

from enum import Enum

import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerLayerNorm
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class LayerNormKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    LIGER = "liger"
    CUSTOM = "custom"


class LayerNormOp(Operation):
    @property
    def name(self) -> str:
        return "layernorm"

    @property
    def kernels(self) -> type[LayerNormKernels]:
        return LayerNormKernels

    def yield_bwd(self, inputs: tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (x, norm_shape, weight, bias, dLdy) = inputs
        match kernel:
            case LayerNormKernels.TCH:
                ref = F.layer_norm(x, norm_shape, weight=weight, bias=bias)
                f = lambda: ref.backward(dLdy, retain_graph=True)
            case LayerNormKernels.CUSTOM:
                ref = MarineLayerNorm.apply(x, norm_shape, weight, bias)
                f = lambda: ref.backward(dLdy, retain_graph=True)
            case LayerNormKernels.TCH_CMP:
                ref = F.layer_norm(x, norm_shape, weight=weight, bias=bias)
                f = torch.compile(
                    lambda: ref.backward(dLdy, retain_graph=True),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(3):
                    f()  # warm up
            case LayerNormKernels.LIGER:
                ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
                ln.weight = torch.nn.Parameter(weight)
                ln.bias = torch.nn.Parameter(bias)
                ref = ln(x)
                f = lambda: ref.backward(dLdy, retain_graph=True)
        return f

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (x, norm_shape, weight, bias) = inputs

        match kernel:
            case LayerNormKernels.TCH:
                f = lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias)
            case LayerNormKernels.CUSTOM:
                f = lambda: MarineLayerNorm.apply(x, norm_shape, weight, bias)
            case LayerNormKernels.TCH_CMP:
                f = torch.compile(
                    lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(5):
                    f()  # warm up
            case LayerNormKernels.LIGER:
                ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
                ln.weight = torch.nn.Parameter(weight)
                ln.bias = torch.nn.Parameter(bias)
                f = lambda: ln(x)

        return f

    def generate_fwd_inputs(self, args: Any):
        (M, N, dtype) = args
        torch.manual_seed(0)

        x = torch.rand((M, N), device=DEVICE, dtype=dtype)
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)

        return (x, norm_shape, weight, bias)

    def generate_bwd_inputs(self, args: Any):
        (x, norm_shape, weight, bias) = self.generate_fwd_inputs(args)
        x.requires_grad_(True)
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        dy = 0.1 * torch.randn_like(x)
        return (x, norm_shape, weight, bias, dy)

    def fwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        # GBPS
        (x, *_) = inputs
        return lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    def bwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        # GBPS
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
