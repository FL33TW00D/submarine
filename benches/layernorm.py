"""
What functionality do we need in here?

We need to be able to benchmark all the variations

"""

from typing import Callable, Any, Tuple, Self, Optional

from enum import Enum

import torch
import torch.nn.functional as F
from benchmark import Benchmark
from liger_kernel.transformers import LigerLayerNorm
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

from marine_ops.marine_ln import MarineLayerNorm


class KernelEnum(Enum):
    """Mixin that provides line_vals and line_names for any Enum."""

    @classmethod
    def line_vals(cls) -> list[str]:
        return [k.value for k in cls]

    @classmethod
    def line_names(cls) -> list[str]:
        return [k.value.title() for k in cls]


class LayerNormKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    LIGER = "liger"
    CUSTOM = "custom"

    def gbps_fwd():
        pass

    def gbps_bwd():
        pass


class LayerNormBenchmark(Benchmark[LayerNormKernels]):
    def yield_bwd(self, inputs: tuple[Any, ...], kernel: LayerNormKernels) -> Callable:
        (x, norm_shape, weight, bias, dLdy) = inputs
        match LayerNormKernels(kernel):
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

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: LayerNormKernels) -> Callable:
        (x, norm_shape, weight, bias) = inputs

        match LayerNormKernels(kernel):
            case LayerNormKernels.TCH:
                f = lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias)
            case LayerNormKernels.CUSTOM:
                f = lambda: MarineLayerNorm.apply(x, norm_shape, weight, bias)
            case LayerNormKernels.TCH_CMP:
                f = torch.compile(
                    lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(3):
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

    def fwd_gbps(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        (x, _) = inputs
        return lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
