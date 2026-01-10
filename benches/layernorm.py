"""
What functionality do we need in here?

We need to be able to benchmark all the variations

"""

from typing import Callable, Any, Tuple

from enum import Enum

import torch
import torch.nn.functional as F
from benchmark import Benchmark
from liger_kernel.transformers import LigerLayerNorm
from triton.testing import do_bench

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


def handle_fwd(kernel, q, x, norm_shape, weight, bias):
    ms, min_ms, max_ms = do_bench(
        f,
        quantiles=q,
        rep=500,
    )
    return ms, min_ms, max_ms


def handle_bwd(kernel, q, x, norm_shape, weight, bias, dLdy):
    x.requires_grad_(True)
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

    ms, min_ms, max_ms = do_bench(f, quantiles=q, rep=500)
    return ms, min_ms, max_ms
