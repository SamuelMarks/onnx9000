"""Backward-compatible re-export module for `onnx9000_onnx_checker`."""

from onnx9000.onnx_checker import (
    SchemaRegistry,
    UnsupportedOpError,
    UnsupportedOpsetError,
    ValidationContext,
    ValidationError,
    check_model,
    check_model_async,
    check_tensor,
    run,
)

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
