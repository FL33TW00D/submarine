from custom_ln import CustomLayerNorm
import enum

import cyclopts
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from liger_kernel.transformers import LigerLayerNorm
from hypothesis import given, settings, strategies as st

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
def validate(max_examples: int = 100, seed: int | None = None):
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
    def check_layernorm(m: int, n: int, dtype: torch.dtype, data_seed: int):
        torch.manual_seed(data_seed)
        x = torch.rand((m, n), device=DEVICE, dtype=dtype)
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=dtype)

        y_custom = CustomLayerNorm.apply(x, norm_shape, weight, bias)
        y_torch = F.layer_norm(x, norm_shape, weight, bias)

        assert torch.allclose(y_custom, y_torch, atol=TOLS[dtype], rtol=TOLS[dtype]), (
            f"Mismatch for shape=({m}, {n}), dtype={dtype}\n"
            f"Max abs diff: {(y_custom - y_torch).abs().max().item():.2e}"
        )

    print(f"Running {max_examples} property-based tests...")
    if seed is not None:
        import random

        random.seed(seed)

    check_layernorm()
    print("✓ All tests passed!")


@app.command
def bench(
    mode: Mode,
    dtype: Dtype,
    m: int = 2048,
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

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[(2**i) - 1 for i in range(8, 15)],
            line_arg="provider",
            line_vals=Kernel.line_vals(),
            line_names=Kernel.line_names(),
            styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
            ylabel="GB/s",
            plot_name=f"layernorm-fwd-{dtype.value}",
            args={"M": m, "mode": mode, "torch_dtype": torch_dtype},
        )
    )
    def benchmark_fn(M, N, provider, mode, torch_dtype):
        x = torch.rand((M, N), device=DEVICE, dtype=torch_dtype)
        q = [0.5, 0.2, 0.8]
        norm_shape = (x.shape[-1],)
        weight = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)
        bias = torch.rand(norm_shape, device=DEVICE, dtype=torch_dtype)

        match Mode(mode):
            case Mode.FORWARD:
                match Kernel(provider):
                    case Kernel.TCH:
                        ms, min_ms, max_ms = triton.testing.do_bench(
                            lambda: F.layer_norm(
                                x, norm_shape, weight=weight, bias=bias
                            ),
                            quantiles=q,
                        )
                    case Kernel.CUSTOM:
                        ms, min_ms, max_ms = triton.testing.do_bench(
                            lambda: CustomLayerNorm.apply(x, norm_shape, weight, bias),
                            quantiles=q,
                        )
                    case Kernel.TCH_CMP:
                        compiled_fn = torch.compile(
                            lambda: F.layer_norm(
                                x, norm_shape, weight=weight, bias=bias
                            ),
                            mode="max-autotune-no-cudagraphs",
                        )
                        compiled_fn()  # warm
                        ms, min_ms, max_ms = triton.testing.do_bench(
                            compiled_fn,
                            quantiles=q,
                        )
                    case Kernel.LIGER:
                        ln = LigerLayerNorm(hidden_size=norm_shape, eps=1e-5)
                        ln.weight = torch.nn.Parameter(weight)
                        ln.bias = torch.nn.Parameter(bias)
                        ms, min_ms, max_ms = triton.testing.do_bench(
                            lambda: ln(x),
                            quantiles=q,
                        )
            case Mode.BACKWARD:
                print("backward")

        def gbps(ms):
            return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    print(f"Running benchmark with dtype={dtype.value}, M={m}")
    benchmark_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)


if __name__ == "__main__":
    app()
