import enum

import cyclopts
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from liger_kernel.transformers import LigerLayerNorm

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


@app.command
def validate(
    dtype: Dtype,
    m: int = 1823,
    n: int = 781,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    seed: int = 0,
):
    """Validate the custom layernorm kernel against torch.

    Args:
        dtype: Data type for the tensors.
        m: Number of rows.
        n: Number of columns.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch_dtype = dtype.to_torch()
    x = torch.randn(m, n, device=DEVICE, dtype=torch_dtype)

    y_triton = our_ln(x)
    y_torch = F.layer_norm(x, (x.shape[-1],))

    if torch.allclose(y_triton, y_torch, atol=atol, rtol=rtol):
        print(f"✓ Validation passed for {dtype.value} with shape ({m}, {n})")
        print(f"  Max absolute diff: {(y_triton - y_torch).abs().max().item():.2e}")
    else:
        print(f"✗ Validation FAILED for {dtype.value} with shape ({m}, {n})")
        print(f"  Max absolute diff: {(y_triton - y_torch).abs().max().item():.2e}")
        print(f"  Triton output sample: {y_triton[0, :5]}")
        print(f"  Torch output sample:  {y_torch[0, :5]}")
        raise SystemExit(1)


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
