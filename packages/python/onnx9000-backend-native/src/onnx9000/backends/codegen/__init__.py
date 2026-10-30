"""Codegen Sub-Package.

Translates ONNX IR graphs into high-performance C++ code via template rendering.
Provides the central Generator class and all registered operator emitters.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.ops import (
    autograd_ops,
    control_flow,
    elementwise,
    math,
    matmul,
    nn,
    sequence,
    shape,
    tensor_ops,
)

__all__ = [
    "Generator",
    "autograd_ops",
    "control_flow",
    "elementwise",
    "math",
    "matmul",
    "nn",
    "sequence",
    "shape",
    "tensor_ops",
]
