"""Apple Backend package."""

from onnx9000.backends.apple.bindings import (
    is_accelerate_available,
    is_metal_available,
    is_mps_available,
)
from onnx9000.backends.apple.executor import Dispatcher

__all__ = ["Dispatcher", "is_accelerate_available", "is_metal_available", "is_mps_available"]
