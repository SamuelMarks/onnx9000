"""Operator registration decorators for ONNX9000 Diffusers."""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)


def register_op(domain: str, op_name: str) -> Callable[[T], T]:
    """Register a custom diffusion operator class under the given ONNX domain and name.

    Args:
        domain: ONNX operator domain string.
        op_name: Operator type name.

    Returns:
        Class decorator attaching `_domain` and `_op_name` metadata.
    """

    def decorator(cls: T) -> T:
        """Attach domain and operator name attributes to the decorated class."""
        cls._domain = domain
        cls._op_name = op_name
        return cls

    return decorator
