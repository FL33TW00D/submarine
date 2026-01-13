from marine_ops.fa import MarineFA
from operations.operation import Operation, KernelEnum
from typing import Callable, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


class FAKernels(KernelEnum):
    TCH = "torch"
    TCH_CMP = "torch_compile"
    CUSTOM = "custom"


class FAOp(Operation):
    @property
    def name(self) -> str:
        return "fa"

    @property
    def kernels(self) -> type[FAKernels]:
        return FAKernels

    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        return lambda f: None

    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable:
        (q, k, v) = inputs

        match kernel:
            case FAKernels.TCH:
                f = lambda: F.scaled_dot_product_attention(q, k, v)
            case FAKernels.CUSTOM:
                f = lambda: MarineFA.apply(q, k, v)
            case FAKernels.TCH_CMP:
                f = torch.compile(
                    lambda: F.scaled_dot_product_attention(q, k, v),
                    mode="max-autotune-no-cudagraphs",
                )
                for _ in range(5):
                    f()  # warm up

        return f

    def generate_fwd_inputs(self, args: Any):
        (B, NH, seq_len, D, dtype) = args
        torch.manual_seed(0)
        q = torch.randn((B, NH, seq_len, D), dtype=dtype, device=DEVICE)
        k = torch.randn((B, NH, seq_len, D), dtype=dtype, device=DEVICE)
        v = torch.randn((B, NH, seq_len, D), dtype=dtype, device=DEVICE)
        return (q, k, v)

    def generate_bwd_inputs(self, args: Any):
        pass

    def fwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        # 𝑜(𝑁2𝑑2𝑀−1)
        (q, k, v, *_) = inputs

        batch, heads, seq_len, head_dim = q.shape
        element_size = q.element_size()

        M = torch.cuda.get_device_properties(DEVICE).shared_memory_per_multiprocessor / q.element_size()
        # HBM accesses: O(N² * d² / M) elements per batch*head
        # Total bytes = batch * heads * (N² * d² / M) * element_size
        hbm_accesses = batch * heads * (seq_len**2) * (head_dim**2) / M * element_size

        return lambda ms: hbm_accesses * 1e-9 / (ms * 1e-3)

    def bwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        return None

    def get_benchmark(self, mode: Any, dtype: Any, **kwargs) -> triton.testing.Benchmark:
        import triton.testing

        b = 2
        nh = 32
        d = 128

        seq_lens = [2**i for i in range(8, 15)]
        seq_lens.insert(-1, 12288)

        y_label = "GB/s"

        return triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=seq_lens,
            line_arg="kernel",
            line_vals=self.kernels.line_vals(),
            line_names=self.kernels.line_names(),
            styles=[("blue", "-"), ("green", "--"), ("red", "-"), ("pink", "--")],
            ylabel=y_label,
            plot_name=f"{self.name}-{mode.value}-{dtype.value}",
            args={"B": b, "NH": nh, "D": d, "mode": mode, "torch_dtype": dtype.to_torch()},
        )

    def dims_to_input_args(self, dims: dict, torch_dtype: Any) -> Tuple[Any, ...]:
        return (dims["B"], dims["NH"], dims["seq_len"], dims["D"], torch_dtype)
