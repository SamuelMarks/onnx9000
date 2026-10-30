"""Backward-compatible re-export module for `onnx9000_custom_ops`."""

from onnx9000.custom_ops import CustomOpFn, CustomOpRegistry, registry

__all__ = ["CustomOpFn", "CustomOpRegistry", "registry"]
