from marine_ops.marine_ln import MarineLayerNorm
import enum

import cyclopts
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from liger_kernel.transformers import LigerLayerNorm
from hypothesis import given, settings, strategies as st
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()

app = cyclopts.App()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

TOLS = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 7e-2}


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


class Kernel(enum.Enum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    LIGER = "liger"
    CUSTOM = "custom"

    @classmethod
    def line_vals(cls) -> list[str]:
        return [k.value for k in cls]

    @classmethod
    def line_names(cls) -> list[str]:
        return [k.value.title() for k in cls]


@app.command
def validate(mode: Mode, max_examples: int = 100, seed: int | None = None):
    """
    Use property based testing to validate custom layernorm against pytorch
    """

    @given(
        m=st.integers(min_value=8, max_value=4096),
        n=st.integers(min_value=8, max_value=4096),
        dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
        data_seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=max_examples, deadline=None, database=None)
    def check_ln_fwd(m: int, n: int, dtype: torch.dtype, data_seed: int):
        torch.manual_seed(data_seed)
        x = torch.rand((m, n), device=DEVICE, dtype=dtype)
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)

        y_custom = MarineLayerNorm.apply(x, norm_shape, weight, bias)
        y_torch = F.layer_norm(x, norm_shape, weight, bias)

        assert torch.allclose(y_custom, y_torch, atol=TOLS[dtype], rtol=TOLS[dtype]), (
            f"Mismatch for shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(y_custom - y_torch).abs().max().item():.2e}"
        )

    @given(
        m=st.integers(min_value=8, max_value=4096),
        n=st.integers(min_value=8, max_value=4096),
        dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
        data_seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=max_examples, deadline=None, database=None)
    def check_ln_bwd(m: int, n: int, dtype: torch.dtype, data_seed: int):
        torch.manual_seed(data_seed)

        x = torch.rand((m, n), device=DEVICE, dtype=dtype)
        x.requires_grad = True
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
        dy = 0.1 * torch.randn_like(x)

        y_custom = MarineLayerNorm.apply(x, norm_shape, weight, bias)
        y_custom.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]

        y_torch = F.layer_norm(x, norm_shape, weight, bias)
        y_torch.backward(dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]

        assert torch.allclose(dx_ref, dx_tri, atol=TOLS[dtype], rtol=TOLS[dtype]), (
            f"Mismatch for DX, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
        )
        assert torch.allclose(dw_ref, dw_tri, atol=TOLS[dtype], rtol=TOLS[dtype]), (
            f"Mismatch for DW, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
        )
        assert torch.allclose(db_ref, db_tri, atol=TOLS[dtype], rtol=TOLS[dtype]), (
            f"Mismatch for DB, shape=({m}, {n}), dtype={dtype}\nMax abs diff: {(dx_ref - dx_tri).abs().max().item():.2e}"
        )

    print(f"Running {max_examples} property-based tests...")
    if seed is not None:
        import random

        random.seed(seed)

    match Mode(mode):
        case Mode.FORWARD:
            check_ln_fwd()
        case Mode.BACKWARD:
            check_ln_bwd()
    print("✓ All tests passed!")


def handle_fwd(kernel, q, x, norm_shape, weight, bias):
    match Kernel(kernel):
        case Kernel.TCH:
            f = lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias)
        case Kernel.CUSTOM:
            f = lambda: MarineLayerNorm.apply(x, norm_shape, weight, bias)
        case Kernel.TCH_CMP:
            f = torch.compile(
                lambda: F.layer_norm(x, norm_shape, weight=weight, bias=bias),
                mode="max-autotune-no-cudagraphs",
            )
            for _ in range(3):
                f()  # warm up
        case Kernel.LIGER:
            ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
            ln.weight = torch.nn.Parameter(weight)
            ln.bias = torch.nn.Parameter(bias)
            f = lambda: ln(x)

    ms, min_ms, max_ms = do_bench(
        f,
        quantiles=q,
        rep=500,
    )
    return ms, min_ms, max_ms


def handle_bwd(kernel, q, x, norm_shape, weight, bias, dLdy):
    x.requires_grad_(True)
    match Kernel(kernel):
        case Kernel.TCH:
            ref = F.layer_norm(x, norm_shape, weight=weight, bias=bias)
            f = lambda: ref.backward(dLdy, retain_graph=True)
        case Kernel.CUSTOM:
            ref = MarineLayerNorm.apply(x, norm_shape, weight, bias)
            f = lambda: ref.backward(dLdy, retain_graph=True)
        case Kernel.TCH_CMP:
            ref = F.layer_norm(x, norm_shape, weight=weight, bias=bias)
            f = torch.compile(
                lambda: ref.backward(dLdy, retain_graph=True),
                mode="max-autotune-no-cudagraphs",
            )
            for _ in range(3):
                f()  # warm up
        case Kernel.LIGER:
            ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
            ln.weight = torch.nn.Parameter(weight)
            ln.bias = torch.nn.Parameter(bias)
            ref = ln(x)
            f = lambda: ref.backward(dLdy, retain_graph=True)

    ms, min_ms, max_ms = do_bench(f, quantiles=q, rep=500)
    return ms, min_ms, max_ms


@app.command
def ncu(
    kernel: Kernel = Kernel.CUSTOM,
    mode: Mode = Mode.FORWARD,
    m: int = 2048,
    n: int = 2048,
    dtype: Dtype = Dtype.BFLOAT16,
):
    """Run NCU profiling for LayerNorm kernels."""
    torch.manual_seed(0)
    torch_dtype = dtype.to_torch()

    x = torch.rand((m, n), device=DEVICE, dtype=torch_dtype)
    norm_shape = (x.shape[-1],)
    weight = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)
    bias = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)

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
    xc = [(2**i) for i in range(8, 15)]
    xc.insert(-1, 12288)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=xc,
            line_arg="provider",
            line_vals=Kernel.line_vals(),
            line_names=Kernel.line_names(),
            styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
            ylabel="GB/s",
            plot_name=f"layernorm-{mode.value}-{dtype.value}",
            args={"M": m, "mode": mode, "torch_dtype": torch_dtype},
        )
    )
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


if __name__ == "__main__":
    app()
