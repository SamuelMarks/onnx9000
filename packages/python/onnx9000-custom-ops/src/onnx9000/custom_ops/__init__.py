"""Custom ONNX operation registration and lookup for ONNX9000."""

from typing import Protocol


class CustomOpFn(Protocol):
    """Protocol for custom ONNX operator functions."""

    def __call__(
        self, *args: float | int | str | bytes, **kwargs: float | int | str | bytes
    ) -> str | int | float | bytes | list[float] | None:
        """Invoke the custom operator implementation."""


class CustomOpRegistry:
    """Registry for custom ONNX operations."""

    def __init__(self) -> None:
        """Initialize an empty custom operation registry."""
        self._ops: dict[str, CustomOpFn] = {}

    def register(self, op_name: str, op_func: CustomOpFn) -> None:
        """Register a custom operation implementation by name.

        Args:
            op_name: Unique name of the custom ONNX operator.
            op_func: Callable implementing the custom operator.
        """
        self._ops[op_name] = op_func

    def get_op(self, op_name: str) -> CustomOpFn | None:
        """Retrieve a registered custom operation by name.

        Args:
            op_name: Name of the custom ONNX operator to look up.

        Returns:
            The registered callable if found, or `None` otherwise.
        """
        return self._ops.get(op_name)

    def list_ops(self) -> list[str]:
        """List all registered custom operation names.

        Returns:
            A list of registered operator names in insertion order.
        """
        return list(self._ops.keys())


registry = CustomOpRegistry()

__all__ = ["CustomOpFn", "CustomOpRegistry", "registry"]
