"""Identity layer."""

from typing import Any
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Tensor


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

        return record_op("Identity", [input])
