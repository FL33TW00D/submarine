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


@triton.jit
def layernorm_fwd_kernel(
    x_ptr,
    output_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    irange = pid * N + offs
    x = tl.load(x_ptr + irange, offs < N, other=0.0)
    mu = tl.sum(x) / N
    x_shift = x - mu
    var = tl.sum(x_shift * x_shift) / N
    x_norm = x_shift * tl.rsqrt(var + eps)
    tl.store(output_ptr + irange, x_norm, offs < N)


def our_ln(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE
    M, N = output.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_fwd_kernel[(M, 1, 1)](x, output, M, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


@given(
    m=st.integers(min_value=1, max_value=4096),
    n=st.integers(min_value=1, max_value=8192),
    dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_layernorm_matches_torch(m: int, n: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)

    x = torch.randn(m, n, device=DEVICE, dtype=dtype)

    y_custom = our_ln(x)
    y_torch = F.layer_norm(x, (x.shape[-1],))

    assert torch.allclose(y_custom, y_torch, atol=1e-5, rtol=1e-5), (
        f"Mismatch for shape=({m}, {n}), dtype={dtype}\n"
        f"Max abs diff: {(y_custom - y_torch).abs().max().item():.2e}"
    )


@app.command
def validate(max_examples: int = 100, seed: int | None = None):
    """
    Use property based testing to validate custom layernorm against pytorch
    """

    @given(
        m=st.integers(min_value=1, max_value=4096),
        n=st.integers(min_value=1, max_value=8192),
        dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
        data_seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=max_examples, deadline=None, database=None)
    def check_layernorm(m: int, n: int, dtype: torch.dtype, data_seed: int):
        torch.manual_seed(data_seed)
        x = torch.randn(m, n, device=DEVICE, dtype=dtype)

        y_custom = our_ln(x)
        y_torch = F.layer_norm(x, (x.shape[-1],))

        assert torch.allclose(y_custom, y_torch, atol=1e-2, rtol=1e-2), (
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
            args={"M": m, "torch_dtype": torch_dtype},
        )
    )
    def benchmark_fn(M, N, provider, torch_dtype):
        x = torch.rand((M, N), device=DEVICE, dtype=torch_dtype)
        quantiles = [0.5, 0.2, 0.8]
        norm_shape = (x.shape[-1],)

        match Kernel(provider):
            case Kernel.TCH:
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: F.layer_norm(x, norm_shape),
                    quantiles=quantiles,
                )
            case Kernel.CUSTOM:
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: our_ln(x),
                    quantiles=quantiles,
                )
            case Kernel.TCH_CMP:
                compiled_fn = torch.compile(
                    lambda: F.layer_norm(x, norm_shape),
                    mode="max-autotune-no-cudagraphs",
                )
                compiled_fn()  # warm
                ms, min_ms, max_ms = triton.testing.do_bench(
                    compiled_fn,
                    quantiles=quantiles,
                )
            case Kernel.LIGER:
                ln = LigerLayerNorm(hidden_size=norm_shape).cuda()
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: ln(x),
                    quantiles=quantiles,
                )

        def gbps(ms):
            return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    print(f"Running benchmark with dtype={dtype.value}, M={m}")
    benchmark_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)


if __name__ == "__main__":
    app()
