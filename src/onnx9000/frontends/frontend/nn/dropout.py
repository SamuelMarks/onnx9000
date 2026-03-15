"""Dropout layers."""

from typing import Any
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Tensor


class _DropoutNd(Module):
    """Provides semantic functionality and verification."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Provides semantic functionality and verification."""
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Provides semantic functionality and verification."""
        from onnx9000.frontends.frontend.utils import record_op

        if not self.training or self.p == 0.0:
            return record_op("Identity", [input])
        res = record_op("Dropout", [input], {"ratio": self.p})
        if isinstance(res, list):
            return res
        return res


class Dropout(_DropoutNd):
    """Dropout layer."""


class Dropout1d(_DropoutNd):
    """Dropout1d layer."""


class Dropout2d(_DropoutNd):
    """Dropout2d layer."""


class Dropout3d(_DropoutNd):
    """Dropout3d layer."""
