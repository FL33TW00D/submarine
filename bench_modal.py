# https://github.com/gau-nernst/learn-cuda/blob/main/12_megakernel/mlp_main.py
import argparse
import importlib
import math
from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import cuda.bench


from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    tflops: float
    membw: float  # GB/s, measured memcpy bandwidth


class GPU(Enum):
    RTX_3090 = GPUSpec(tflops=142, membw=836)
    RTX_4090 = GPUSpec(tflops=330, membw=1008)
    RTX_5090 = GPUSpec(tflops=209.5, membw=1500)
    A100 = GPUSpec(tflops=312, membw=1700)
    H100 = GPUSpec(tflops=979, membw=3350)
    H200 = GPUSpec(tflops=1979, membw=4000)
    B200 = GPUSpec(tflops=4500, membw=8000)


def detect_gpu() -> GPU:
    gpu_name = torch.cuda.get_device_name()
    match gpu_name:
        case name if "3090" in name:
            return GPU.RTX_3090
        case name if "4090" in name:
            return GPU.RTX_4090
        case name if "5090" in name:
            return GPU.RTX_5090
        case name if "A100" in name:
            return GPU.A100
        case name if "H100" in name:
            return GPU.H100
        case name if "H200" in name:
            return GPU.H200
        case name if "B200" in name:
            return GPU.B200
        case _:
            raise ValueError(f"Unknown GPU: {gpu_name}")


def fused_linear_cross_entropy_ref(x: Tensor, weight: Tensor, target: Tensor) -> torch.Tensor:
    logits = x @ weight.T  # (BT, V)
    return F.cross_entropy(logits.to(torch.float32), target, reduction="none").to(torch.float32)


def get_sol() -> tuple[float, float]:
    gpu = detect_gpu()
    return gpu.value.tflops, gpu.value.membw


def get_kernel(name: str):
    if name == "eager":
        f = fused_linear_cross_entropy_ref
    elif name == "inductor":
        f = torch.compile(
            fused_linear_cross_entropy_ref, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=True
        )
    else:
        m_name, f_name = name.rsplit(".", 1)
        f = getattr(importlib.import_module(m_name), f_name)
    return f


def to_torch_stream(s: "cuda.bench.CudaStream", device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: "cuda.bench.State") -> None:
    device = state.get_device()

    f = get_kernel(state.get_string("kernel"))

    BT = state.get_int64("BT")
    D = state.get_int64("D")
    V = state.get_int64("V")

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        x = torch.randn(BT, D, dtype=torch.bfloat16, device=device)
        weight = torch.randn(V, D, dtype=torch.bfloat16, device=device)
        target = torch.randint(0, V, (BT,), device=device)

        out_ref = fused_linear_cross_entropy_ref(x, weight, target)
        out = f(x, weight, target)
        # torch.testing.assert_close(out, out_ref)

        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            # apply scaling to make sure the output doesn't explode
            x = torch.randn(BT, D, dtype=torch.bfloat16, device=device)
            weight = torch.randn(V, D, dtype=torch.bfloat16, device=device)
            target = torch.randint(0, V, (BT,), device=device)
            inputs_list.append((x, weight, target))

    def launcher(launch: "cuda.bench.Launch") -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for x, weight, target in inputs_list:
                f(x, weight, target)

    state.exec(launcher, sync=True)


def benchmark(shape: list[int]):
    import cuda.bench

    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")

    BT, D, V = shape

    # we also count writing and reading tmp buffer in total memory traffic
    gemm_flops = 2 * BT * D * V
    non_gemm_flops = 5 * BT * V
    num_flops = gemm_flops + non_gemm_flops
    num_gb = ((BT * D + V * D) * 2 + 2 * BT * 4) * 1e-9

    # duplicate inputs to make sure each measurement is at least 10ms
    SOL_COMPUTE, SOL_MEMORY = get_sol()
    min_compute_latency_ms = num_flops / (SOL_COMPUTE * 1e12) * 1e3
    min_memory_latency_ms = num_gb / SOL_MEMORY * 1e3
    min_latency_ms = max(min_compute_latency_ms, min_memory_latency_ms)
    num_inputs = min(math.ceil(10 / min_latency_ms), 1000)

    kernels_list = []
    kernels_list += ["eager", "inductor"]
    kernels_list += ["marine_ops.fused_cross_entropy.fused_linear_cross_entropy"]

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("kernel", kernels_list)
    bench.add_int64_axis("BT", [BT])
    bench.add_int64_axis("D", [D])
    bench.add_int64_axis("V", [V])
    bench.add_int64_axis("num_inputs", [num_inputs])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= num_inputs  # rescale
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["TFLOPS"] = num_flops / df["GPU Time (sec)"] * 1e-12
    df["membw (GB/s)"] = num_gb / df["GPU Time (sec)"]

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["membw (GB/s)"] = df["membw (GB/s)"].map("{:.2f}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "TFLOPS", "membw (GB/s)"]].to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs="+", default=[16384, 1024, 131072])
    parser.add_argument("--modal")
    args = parser.parse_args()

    if args.modal is None:
        benchmark(args.shape)
    else:
        import modal

        image = (
            modal.Image.from_registry("nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12")
            .entrypoint([])  # remove verbose logging by base image on entry
            .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
            .uv_pip_install("ninja", "pandas", "tabulate", "cuda-bench[cu13]")
            .add_local_python_source("marine_ops")
        )
        app = modal.App("fused-xentropy", image=image)
        modal_main = app.function(image=image, gpu=args.modal)(benchmark)

        with modal.enable_output(), app.run():
            modal_main.remote(args.shape)
