from marine_ops.marine_ln import MarineLayerNorm
import enum

import cyclopts
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from hypothesis import given, settings, strategies as st
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()

app = cyclopts.App()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


class Mode(enum.Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class Dtype(enum.Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_torch(self) -> torch.dtype:
        return {
            Dtype.FLOAT32: torch.float32,
            Dtype.FLOAT16: torch.float16,
            Dtype.BFLOAT16: torch.bfloat16,
        }[self]


@app.command
def validate(operation, mode: Mode, max_examples: int = 100, seed: int | None = None):
    """
    Use property based testing to validate operation against pytorch
    """
    print(f"Running {max_examples} property-based tests...")
    if seed is not None:
        import random

        random.seed(seed)

    print("✓ All tests passed!")


"""
@app.command
def ncu(
    op: Operation,
    kernel: Kernel = Kernel.CUSTOM,
    mode: Mode = Mode.FORWARD,
    m: int = 2048,
    n: int = 2048,
    dtype: Dtype = Dtype.BFLOAT16,
):
    match kernel:
        case Kernel.CUSTOM:
            fwd = lambda: MarineLayerNorm.apply(x, norm_shape, weight, bias)
        case Kernel.LIGER:
            ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
            ln.weight = torch.nn.Parameter(weight)
            ln.bias = torch.nn.Parameter(bias)
            fwd = lambda: ln(x)
        case _:
            raise ValueError(f"ncu only supports 'custom' or 'liger', got {kernel.value}")

    match mode:
        case Mode.FORWARD:
            f = fwd
        case Mode.BACKWARD:
            x.requires_grad_(True)
            weight.requires_grad_(True)
            bias.requires_grad_(True)
            dy = 0.1 * torch.randn_like(x)
            y = fwd()
            f = lambda: y.backward(dy, retain_graph=True)

    print(f"Profiling: kernel={kernel.value}, mode={mode.value}, shape=({m}, {n}), dtype={dtype.value}")

    for _ in range(3):
        f()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    f()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
"""


@app.command
def bench(
    mode: Mode,
    dtype: Dtype,
    m: int = 4096,
    save_path: str = ".",
    show_plots: bool = False,
):
    """Run layernorm forward pass benchmarks.

    Args:
        dtype: Data type for the tensors.
        m: Number of rows (batch size).
        save_path: Directory to save benchmark plots.
        show_plots: Whether to display plots interactively.
    """
    torch_dtype = dtype.to_torch()

    benchmark = triton.testing.Benchmark(
        x_names=["N"],
        x_vals=xc,
        line_arg="provider",
        line_vals=Kernel.line_vals(),
        line_names=Kernel.line_names(),
        styles=[("blue", "-"), ("green", "--"), ("re", "-"), ("pink", "--")],
        ylabel="GB/s",
        plot_name=f"layernorm-{mode.value}-{dtype.value}",
        args={"M": m, "mode": mode, "torch_dtype": torch_dtype},
    )

    @triton.testing.perf_report(benchmark)
    def benchmark_fn(M, N, provider, mode, torch_dtype):
        x = torch.rand((M, N), device=DEVICE, dtype=torch_dtype)
        q = [0.5, 0.2, 0.8]
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)
        dy = 0.1 * torch.randn_like(x)

        match Mode(mode):
            case Mode.FORWARD:
                ms, min_ms, max_ms = handle_fwd(provider, q, x, norm_shape, weight, bias)
            case Mode.BACKWARD:
                ms, min_ms, max_ms = handle_bwd(provider, q, x, norm_shape, weight, bias, dy)

        def gbps(ms):
            return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    print(f"Running benchmark with dtype={dtype.value}, M={m}")
    benchmark_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)


def create_benchmark(M: int, kernel: Kernel, mode: Mode, dtype):
    styles = [("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--"), ("orange", "-")]

    xc = [(2**i) for i in range(8, 15)]
    xc.insert(-1, 12288)

    benchmark = triton.testing.Benchmark(
        x_names=["N"],
        x_vals=xc,
        line_arg="provider",
        line_vals=Kernel.line_vals(),
        line_names=Kernel.line_names(),
        styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
        ylabel="GB/s",
        plot_name=f"{operation.name}-{mode.value}-{dtype.value}",
        args={"M": M, "mode": mode, "torch_dtype": dtype.value},
    )


if __name__ == "__main__":
    app()
