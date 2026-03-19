"""Codegen Sub-Package.

Translates ONNX IR graphs into high-performance C++ code via template rendering.
Provides the central Generator class and all registered operator emitters.
"""

import onnx9000.backends.codegen.ops.autograd_ops
import onnx9000.backends.codegen.ops.control_flow
import onnx9000.backends.codegen.ops.elementwise
import onnx9000.backends.codegen.ops.math
import onnx9000.backends.codegen.ops.matmul
import onnx9000.backends.codegen.ops.nn
import onnx9000.backends.codegen.ops.sequence
import onnx9000.backends.codegen.ops.shape
import onnx9000.backends.codegen.ops.tensor_ops
from onnx9000.backends.codegen.generator import Generator

__all__ = ["Generator"]
