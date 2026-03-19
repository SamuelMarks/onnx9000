"""Identity layer."""

from typing import Any

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Implement the __init__ method."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Implement the forward method."""
        from onnx9000.converters.frontend.utils import record_op

        return record_op("Identity", [input])
