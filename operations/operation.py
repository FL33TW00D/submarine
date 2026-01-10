from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Callable, Any, Tuple, Optional

K = TypeVar("K", bound=Enum)

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


class Operation(ABC, Generic[K]):
    @property
    @abstractmethod
    def kernel_cls(self) -> type[K]: ...

    @abstractmethod
    def generate_fwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def generate_bwd_inputs(self, args: Any) -> Tuple[Any, ...]: ...

    @abstractmethod
    def yield_fwd(self, inputs: Tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def yield_bwd(self, inputs: Tuple[Any, ...], kernel: K) -> Callable: ...

    @abstractmethod
    def fwd_gbps(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]: ...

    @abstractmethod
    def bwd_gbps(self, inputs: Tuple[Any, ...]) -> Optional[Callable[[int], float]]: ...
