"""
Codegen Sub-Package

Translates ONNX IR graphs into high-performance C++ code via template rendering.
Provides the central Generator class and all registered operator emitters.
"""

import onnx9000.codegen.ops.autograd_ops  # noqa: F401
import onnx9000.codegen.ops.control_flow  # noqa: F401

# Import all ops to register them
import onnx9000.codegen.ops.elementwise  # noqa: F401
import onnx9000.codegen.ops.math  # noqa: F401
import onnx9000.codegen.ops.matmul  # noqa: F401
import onnx9000.codegen.ops.nn  # noqa: F401
import onnx9000.codegen.ops.sequence  # noqa: F401
import onnx9000.codegen.ops.shape  # noqa: F401
import onnx9000.codegen.ops.tensor_ops  # noqa: F401
from onnx9000.codegen.generator import Generator

__all__ = ["Generator"]
