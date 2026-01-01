import triton
import torch
import torch.nn.functional as F
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def layernorm_kernel(
    x_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    irange = pid * N + offs
    x = tl.load(x_ptr + irange, offs < N, other=0.0)

    mu = tl.sum(x) / N
    var = tl.sum((x - mu) * (x - mu)) / N

    x_norm = (x - mu) * tl.rsqrt(var + 1e-5)
    tl.store(output_ptr + irange, x_norm, offs < N)


def layernorm(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE

    M, N = output.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_kernel[(M, 1, 1)](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = layernorm(x)
y_torch = F.layer_norm(x, (x.shape[-1],))
assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), (y_triton, y_torch)

# print PTX out
if True:
    device = torch.cuda.current_device()
    cache_tuple = layernorm_kernel.device_caches[device]
    cache_dict = cache_tuple[0]  # The actual cache dictionary
    compiled_kernel = list(cache_dict.values())[0]
    print(compiled_kernel.asm["ptx"])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 20)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
        ],  # possible values for `line_arg``
        line_names=["Triton", "Torch"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="layernorm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.rand((M, N), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.layer_norm(x, (x.shape[-1],)),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layernorm(x), quantiles=quantiles
        )

    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, save_path=".")
