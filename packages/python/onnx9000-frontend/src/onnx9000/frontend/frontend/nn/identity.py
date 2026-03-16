"""Identity layer."""

from typing import Any
from onnx9000.frontend.frontend.nn.module import Module
from onnx9000.frontend.frontend.tensor import Tensor


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Implements the __init__ method."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Implements the forward method."""
        from onnx9000.frontend.frontend.utils import record_op

        return record_op("Identity", [input])
