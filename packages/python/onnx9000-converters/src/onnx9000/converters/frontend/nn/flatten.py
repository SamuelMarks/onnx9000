"""Flatten layers."""

from typing import Union

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class Flatten(Module):
    """Flatten layer."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        return input.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    """Unflatten layer."""

    def __init__(self, dim: Union[int, str], unflattened_size: Union[tuple[int], list]) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        return input.reshape(list(self.unflattened_size))
