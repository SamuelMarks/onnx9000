"""
Frontend Sub-Package

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from typing import Any, Optional, Union

from onnx9000.dtypes import DType


class Node:
    """Represents an operation in the graph."""

    def __init__(
        self,
        op_type: str,
        inputs: list[Any],
        outputs: list[Any],
        attributes: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the frontend builder or trace context."""

        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}
        self.name = name or ""


class Tensor:
    """Symbolic representation of data in the frontend graph."""

    _id_counter = 0

    def __init__(
        self,
        shape: tuple[Union[int, str], ...],
        dtype: DType,
        name: Optional[str] = None,
        data: Optional[Any] = None,
    ) -> None:
        """Initializes the frontend builder or trace context."""

        self._shape = shape
        self._dtype = dtype
        self.data = data

        if name is None:
            Tensor._id_counter += 1
            self._name = f"tensor_{Tensor._id_counter}"
        else:
            self._name = name

    @property
    def shape(self) -> tuple[Union[int, str], ...]:
        """shape docstring."""

        return self._shape

    @property
    def dtype(self) -> DType:
        """dtype docstring."""

        return self._dtype

    @property
    def name(self) -> str:
        """name docstring."""

        return self._name

    def __repr__(self) -> str:
        """__repr__ docstring."""

        return f"Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"  # pragma: no cover

    def __add__(self, other: "Tensor") -> "Tensor":
        """__add__ docstring."""

        from onnx9000.frontend.utils import record_op

        return record_op("Add", [self, other])

    def __sub__(self, other: "Tensor") -> "Tensor":
        """__sub__ docstring."""

        from onnx9000.frontend.utils import record_op  # pragma: no cover

        return record_op("Sub", [self, other])  # pragma: no cover

    def __mul__(self, other: "Tensor") -> "Tensor":
        """__mul__ docstring."""

        from onnx9000.frontend.utils import record_op

        return record_op("Mul", [self, other])

    def __truediv__(self, other: "Tensor") -> "Tensor":
        """__truediv__ docstring."""

        from onnx9000.frontend.utils import record_op  # pragma: no cover

        return record_op("Div", [self, other])  # pragma: no cover

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """__matmul__ docstring."""

        from onnx9000.frontend.utils import record_op

        return record_op("MatMul", [self, other])


class Parameter(Tensor):
    """Subclass of Tensor denoting trainable weights or fixed initializers."""

    def __init__(
        self,
        shape: tuple[Union[int, str], ...],
        dtype: DType,
        name: Optional[str] = None,
        data: Optional[Any] = None,
    ) -> None:
        """Initializes the frontend builder or trace context."""

        super().__init__(shape, dtype, name)
        self.data = data
