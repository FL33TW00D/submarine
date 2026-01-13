from operations.fa import FAOp
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
    FA = "fa"


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
    save_path: str = ".",
    show_plots: bool = False,
    dump_ptx: bool = False,
):
    match op:
        case OpList.LAYERNORM:
            operation = LayerNormOp()
        case OpList.SOFTMAX:
            operation = SoftmaxOp()
        case OpList.GEMM:
            operation = GEMMOp()
        case OpList.FA:
            operation = FAOp()

    benchmark_config = operation.get_benchmark(mode, dtype)

    @triton.testing.perf_report(benchmark_config)
    def benchmark_fn(kernel, mode, torch_dtype, **dims):
        q = [0.5, 0.2, 0.8]

        input_args = operation.dims_to_input_args(dims, torch_dtype)

        match Mode(mode):
            case Mode.FORWARD:
                inputs = operation.generate_fwd_inputs(input_args)
                f = operation.yield_fwd(inputs, operation.kernels(kernel))
                metric_f = operation.fwd_metric(inputs)
            case Mode.BACKWARD:
                inputs = operation.generate_bwd_inputs(input_args)
                f = operation.yield_bwd(inputs, operation.kernels(kernel))
                metric_f = operation.bwd_metric(inputs)

        ms, min_ms, max_ms = do_bench(f, quantiles=q, rep=500)
        return metric_f(ms), metric_f(min_ms), metric_f(max_ms)

    print(f"Running benchmark: op={op.value}, mode={mode.value}, dtype={dtype.value}")
    benchmark_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)


@app.command
def ncu(
    op: OpList,
    kernel: str,
    mode: Mode = Mode.FORWARD,
    dtype: Dtype = Dtype.BFLOAT16,
):
    """Run NCU profiling for kernel operations using default dimensions."""
    torch.manual_seed(0)
    torch_dtype = dtype.to_torch()

    match op:
        case OpList.LAYERNORM:
            operation = LayerNormOp()
            dims = {"M": 4096, "N": 8192}
        case OpList.SOFTMAX:
            operation = SoftmaxOp()
            dims = {"M": 4096, "N": 8192}
        case OpList.GEMM:
            operation = GEMMOp()
            dims = {"M": 4096, "N": 8192}
        case OpList.FA:
            operation = FAOp()
            dims = {"B": 4, "NH": 8, "seq_len": 4096, "D": 64}

    kernel_enum = operation.kernels(kernel)
    input_args = operation.dims_to_input_args(dims, torch_dtype)

    match mode:
        case Mode.FORWARD:
            inputs = operation.generate_fwd_inputs(input_args)
            f = operation.yield_fwd(inputs, kernel_enum)
        case Mode.BACKWARD:
            inputs = operation.generate_bwd_inputs(input_args)
            f = operation.yield_bwd(inputs, kernel_enum)

    dims_str = ", ".join(f"{k}={v}" for k, v in dims.items())
    print(f"Profiling: op={op.value}, kernel={kernel}, mode={mode.value}, dims=({dims_str}), dtype={dtype.value}")

    for _ in range(3):
        f()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    f()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    app()
