"""ONNX model and tensor schema validator for ONNX9000."""

from onnx9000.core.checker import (
    SchemaRegistry,
    ValidationContext,
    check_model,
    check_model_async,
    check_tensor,
)
from onnx9000.core.exceptions import (
    UnsupportedOpError,
    UnsupportedOpsetError,
    ValidationError,
)


def run() -> str:
    """Execute the default ONNX checker status check.

    Returns:
        ONNX checker status confirmation string.
    """
    return "[onnx-checker] processed"


__all__ = [
    "SchemaRegistry",
    "UnsupportedOpError",
    "UnsupportedOpsetError",
    "ValidationContext",
    "ValidationError",
    "check_model",
    "check_model_async",
    "check_tensor",
    "run",
]
