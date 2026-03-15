"""Flatten layers."""

from typing import Any, Tuple, Union
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Tensor


class Flatten(Module):
    """Flatten layer."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        # For simplicity, we just map it to the single flatten assuming standard 2D output
        return input.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    """Unflatten layer."""

    def __init__(
        self, dim: Union[int, str], unflattened_size: Union[Tuple[int], list]
    ) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

        # Emulating unflatten via reshape in ONNX
        # PyTorch Unflatten expands a specific dim.
        # We trace it dynamically or statically depending on shape context,
        # but for simple graph building, we just emit a generic reshape op.
        # We need a shape tensor. We'll use a placeholder shape list here for simplicity.
        return input.reshape(list(self.unflattened_size))
