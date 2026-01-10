from submarine import Mode
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Callable, Any, Tuple, Optional
from dataclasses import dataclass
from triton.testing import do_bench
import triton


@dataclass
class BenchResult:
    ms: float
    min_ms: float
    max_ms: float


K = TypeVar("K", bound=Enum)


class Benchmark(ABC, Generic[K]):
    quantiles: tuple[float, ...] = (0.5, 0.2, 0.8)
    rep: int = 500
    x_name: str = "N"
    ylabel: str = "GB/s"

    @property
    @abstractmethod
    def kernel_cls(self) -> type[K]: ...

    @abstractmethod
    def x_vals(self) -> list[int]: ...

    def plot_name(self, mode: str, **kwargs) -> str:
        return f"{self.__class__.__name__.lower()}-{mode}"

    @abstractmethod
    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def generate_fwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def generate_bwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def fwd_gbps(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]: ...

    @abstractmethod
    def bwd_gbps(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]: ...

    def run(
        self,
        mode: Mode,
        fixed_args: dict[str, Any],
        save_path: str = ".",
        show_plots: bool = False,
    ):
        kernel_cls = self.kernel_cls
        is_fwd = mode == "fwd"

        generate = self.generate_fwd_inputs if is_fwd else self.generate_bwd_inputs
        yield_fn = self.yield_fwd if is_fwd else self.yield_bwd
        gbps_fn = self.fwd_gbps if is_fwd else self.bwd_gbps

        benchmark = triton.testing.Benchmark(
            x_names=[self.x_name],
            x_vals=self.x_vals(),
            line_arg="kernel",
            line_vals=kernel_cls.line_vals(),
            line_names=kernel_cls.line_names(),
            styles=kernel_cls.styles(),
            ylabel=self.ylabel,
            plot_name=self.plot_name(mode, **fixed_args),
            args={"mode": mode, **fixed_args},
        )

        bench_instance = self

        @triton.testing.perf_report([benchmark])
        def bench_fn(kernel: str, **args):
            N = args[bench_instance.x_name]
            input_args = (args["M"], N, args["dtype"])

            inputs = generate(input_args)
            kernel_enum = kernel_cls(kernel)
            fn = yield_fn(inputs, kernel_enum)

            ms, min_ms, max_ms = do_bench(
                fn,
                quantiles=bench_instance.quantiles,
                rep=bench_instance.rep,
            )

            gbps = gbps_fn(inputs)
            if gbps:
                return gbps(ms), gbps(max_ms), gbps(min_ms)
            return ms, max_ms, min_ms

        bench_fn.run(print_data=True, save_path=save_path, show_plots=show_plots)
