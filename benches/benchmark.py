from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Callable, Any, Tuple, Optional
from dataclasses import dataclass
from triton.testing import do_bench


@dataclass
class BenchResult:
    ms: float
    min_ms: float
    max_ms: float

    def gbps(self, num_bytes: int) -> float:
        return num_bytes / (self.ms * 1e-3) / 1e9


K = TypeVar("K", bound=Enum)


class Benchmark(ABC, Generic[K]):
    quantiles: tuple[float, ...] = (0.5, 0.2, 0.8)
    rep: int = 500

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

    def bench_fwd(self, args: Any, kernel: K) -> Tuple[float | int, float | int, float | int]:
        inputs = self.generate_fwd_inputs(args)
        ms, min_ms, max_ms = do_bench(
            self.yield_fwd(inputs, kernel),
            quantiles=self.quantiles,
            rep=self.rep,
        )

        if self.fwd_gbps(inputs):
            gbps = self.fwd_gbps(inputs)
            return gbps(ms), gbps(min_ms), gbps(max_ms)  # ty:ignore[call-non-callable]
        else:
            return ms, min_ms, max_ms

    def bench_bwd(self, args: Any, kernel: K) -> Tuple[int, int, int]:
        inputs = self.generate_bwd_inputs(args)
        ms, min_ms, max_ms = do_bench(
            self.yield_bwd(inputs, kernel),
            quantiles=self.quantiles,
            rep=self.rep,
        )
        return ms, min_ms, max_ms
