"""Dropout layers."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor


class _DropoutNd(Module):
    """Class _DropoutNd implementation."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize _DropoutNd with configured parameters and state.

        Args:
            p: Parameter specifying p (float).
            inplace: Parameter specifying inplace (bool).
        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """Execute forward on _DropoutNd.

        Args:
            input: Parameter specifying input (Tensor).

        Returns:
            Result of forward (Tensor).
        """
        from onnx9000.converters.frontend.utils import record_op

        if not self.training or self.p == 0.0:
            return record_op("Identity", [input])
        res = record_op("Dropout", [input], {"ratio": self.p})
        if isinstance(res, list):
            return res
        return res


class Dropout(_DropoutNd):
    """Dropout layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize Dropout with configured parameters and state.

        Args:
            p: Parameter specifying p (float).
            inplace: Parameter specifying inplace (bool).
        """
        super().__init__(p=p, inplace=inplace)  # pragma: no cover  # pragma: no cover


class Dropout1d(_DropoutNd):
    """Dropout1d layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize Dropout1d with configured parameters and state.

        Args:
            p: Parameter specifying p (float).
            inplace: Parameter specifying inplace (bool).
        """
        super().__init__(p=p, inplace=inplace)  # pragma: no cover  # pragma: no cover


class Dropout2d(_DropoutNd):
    """Dropout2d layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize Dropout2d with configured parameters and state.

        Args:
            p: Parameter specifying p (float).
            inplace: Parameter specifying inplace (bool).
        """
        super().__init__(p=p, inplace=inplace)  # pragma: no cover  # pragma: no cover


class Dropout3d(_DropoutNd):
    """Dropout3d layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize Dropout3d with configured parameters and state.

        Args:
            p: Parameter specifying p (float).
            inplace: Parameter specifying inplace (bool).
        """
        super().__init__(p=p, inplace=inplace)  # pragma: no cover  # pragma: no cover
