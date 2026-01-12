from abc import ABC, abstractmethod
from typing import Callable, Any, Tuple, Optional
import enum


class KernelEnum(enum.Enum):
    """Mixin that provides line_vals and line_names for any Enum."""

    @classmethod
    def line_vals(cls) -> list[str]:
        return [k.value for k in cls]

    @classmethod
    def line_names(cls) -> list[str]:
        return [k.value.title() for k in cls]


"""
An operation generalizes kernels.

One to many relationship between operation and kernels.
e.g:
LayerNorm
    |
    |--> Liger
    |--> Torch 
    |--> Torch Compile 
    |--> Marine 
"""


class Operation(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def memory_bound(self) -> bool: ...

    @property
    @abstractmethod
    def kernels(self) -> type[KernelEnum]: ...

    @abstractmethod
    def generate_fwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def generate_bwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable: ...

    @abstractmethod
    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: KernelEnum) -> Callable: ...

    def fwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        return None

    def bwd_metric(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]:
        return None
