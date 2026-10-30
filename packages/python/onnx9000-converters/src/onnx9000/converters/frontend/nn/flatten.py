"""Flatten layers."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class Flatten(Module):
    """Flatten layer."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Initialize Flatten with configured parameters and state.

        Args:
            start_dim: Parameter specifying start dim (int).
            end_dim: Parameter specifying end dim (int).
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on Flatten.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        return input.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    """Unflatten layer."""

    def __init__(self, dim: int | str, unflattened_size: tuple[int] | list) -> None:
        """Initialize Unflatten with configured parameters and state.

        Args:
            dim: Parameter specifying dim (int | str).
            unflattened_size: Parameter specifying unflattened size (tuple[int] | list).
        """
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on Unflatten.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        return input.reshape(list(self.unflattened_size))
