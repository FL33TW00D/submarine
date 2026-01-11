from operations.gemm import GEMMOp
from operations.softmax import SoftmaxOp
from operations.layernorm import LayerNormOp
import enum

import cyclopts
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()

app = cyclopts.App()

tch_to_trt = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


class OpList(enum.Enum):
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    GEMM = "gemm"


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
def bench(
    op: OpList,
    mode: Mode,
    dtype: Dtype,
    m: int = 4096,
    save_path: str = ".",
    show_plots: bool = False,
    dump_ptx: bool = False,
):
    torch_dtype = dtype.to_torch()
    xc = [(2**i) for i in range(8, 15)]
    xc.insert(-1, 12288)

    match op:
        case OpList.LAYERNORM:
            operation = LayerNormOp()
        case OpList.SOFTMAX:
            operation = SoftmaxOp()
        case OpList.GEMM:
            operation = GEMMOp()

    y_label = "GB/s" if operation.memory_bound else "ms"

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=xc,
            line_arg="kernel",
            line_vals=operation.kernels.line_vals(),
            line_names=operation.kernels.line_names(),
            styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
            ylabel=y_label,
            plot_name=f"{operation.name}-{mode.value}-{dtype.value}",
            args={"M": m, "mode": mode, "torch_dtype": torch_dtype},
        )
    )
    def benchmark_fn(M, N, kernel, mode, torch_dtype):
        q = [0.5, 0.2, 0.8]

        match Mode(mode):
            case Mode.FORWARD:
                inputs = operation.generate_fwd_inputs((M, N, torch_dtype))
                f = operation.yield_fwd(inputs, operation.kernels(kernel))
                gbps_f = operation.fwd_gbps(inputs)
            case Mode.BACKWARD:
                inputs = operation.generate_bwd_inputs((M, N, torch_dtype))
                f = operation.yield_bwd(inputs, operation.kernels(kernel))
                gbps_f = operation.bwd_gbps(inputs)

        ms, min_ms, max_ms = do_bench(f, quantiles=q, rep=500)
        if gbps_f:
            return gbps_f(ms), gbps_f(max_ms), gbps_f(min_ms)
        else:
            return ms, min_ms, max_ms

    print(f"Running benchmark with dtype={dtype.value}, M={m}")
    benchmark_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)


@app.command
def ncu(
    op: OpList,
    kernel: str,
    mode: Mode = Mode.FORWARD,
    m: int = 2048,
    n: int = 2048,
    dtype: Dtype = Dtype.BFLOAT16,
):
    """Run NCU profiling for kernel operations."""
    torch.manual_seed(0)
    torch_dtype = dtype.to_torch()

    match op:
        case OpList.LAYERNORM:
            operation = LayerNormOp()
        case OpList.SOFTMAX:
            operation = SoftmaxOp()
        case OpList.GEMM:
            operation = GEMMOp()

    kernel_enum = operation.kernels(kernel)
    print(kernel_enum)

    match mode:
        case Mode.FORWARD:
            inputs = operation.generate_fwd_inputs((m, n, torch_dtype))
            f = operation.yield_fwd(inputs, kernel_enum)
        case Mode.BACKWARD:
            inputs = operation.generate_bwd_inputs((m, n, torch_dtype))
            f = operation.yield_bwd(inputs, kernel_enum)

    print(f"Profiling: op={op.value}, kernel={kernel}, mode={mode.value}, shape=({m}, {n}), dtype={dtype.value}")

    for _ in range(3):
        f()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    f()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    app()
