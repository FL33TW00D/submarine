import triton
import torch
import torch.nn.functional as F
import triton.language as tl
from liger_kernel.transformers import LigerLayerNorm
import enum

DEVICE = triton.runtime.driver.active.get_active_torch_device()


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


def our_ln(x: torch.Tensor, eps: float = 1e-5):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE

    M, N = output.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_fwd_kernel[(M, 1, 1)](x, output, M, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = our_ln(x)
y_torch = F.layer_norm(x, (x.shape[-1],))
assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), (y_triton, y_torch)

# print PTX out
if False:
    device = torch.cuda.current_device()
    cache_tuple = layernorm_fwd_kernel.device_caches[device]
    cache_dict = cache_tuple[0]  # The actual cache dictionary
    compiled_kernel = list(cache_dict.values())[0]
    print(compiled_kernel.asm["ptx"])


class Kernel(enum.Enum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    LIGER = "liger"
    CUSTOM = "custom"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            (2**i) - 1 for i in range(8, 15)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "triton",
            "torch_compile",
            "liger",
        ],
        line_names=["Triton", "Torch Eager", "Torch Compile", "Liger"],
        styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
        ylabel="GB/s",  # label name for the y-axis
        plot_name="layernorm-fwd-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 2048},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.rand((M, N), device=DEVICE, dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    norm_shape = (x.shape[-1],)

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.layer_norm(x, norm_shape),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: our_ln(x),
            quantiles=quantiles,
        )
    elif provider == "torch_compile":
        # must use max-autotune-no-cudagraphs https://x.com/gaunernst/status/2005524873477812298?s=20
        compiled_fn = torch.compile(
            lambda: F.layer_norm(x, norm_shape), mode="max-autotune-no-cudagraphs"
        )
        compiled_fn()  # warm
        ms, min_ms, max_ms = triton.testing.do_bench(
            compiled_fn,
            quantiles=quantiles,
        )
    elif provider == "liger":
        ln = LigerLayerNorm(hidden_size=norm_shape).cuda()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ln(x),
            quantiles=quantiles,
        )

    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, save_path=".")
