from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Callable, Any, Tuple
from dataclasses import dataclass
import torch
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
    def yield_fwd(self, inputs: tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def yield_bwd(self, inputs: tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def generate_fwd_inputs(self) -> Tuple[Any, ...]: ...

    def bench_fwd(self, kernel: K) -> Tuple[int, int, int]:
        inputs = self.generate_fwd_inputs()
        f = self.yield_fwd(inputs, kernel)
        ms, min_ms, max_ms = do_bench(
            f,
            quantiles=self.quantiles,
            rep=self.rep,
        )
        return ms, min_ms, max_ms
