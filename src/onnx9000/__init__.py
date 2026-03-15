"""Module providing core logic and structural definitions."""

from onnx9000.core import ops, parser
from onnx9000.core.dtypes import DType
from onnx9000.export.builder import to_onnx, to_string
from onnx9000.frontends.frontend.builder import GraphBuilder, Tracing

# Import `jit` as `jit_decorator` so there is no name collision internally.
from onnx9000.frontends.frontend.jit import jit as jit_decorator
from onnx9000.frontends.frontend.tensor import Node, Parameter, Tensor
from onnx9000.frontends.jit import compile as _compile
from onnx9000.utils.cache import clear_cache

# Explicit re-bind
jit = jit_decorator
compile = _compile

__version__ = "0.0.1"

__all__ = [
    "compile",
    "clear_cache",
    "DType",
    "Tensor",
    "Parameter",
    "Node",
    "GraphBuilder",
    "Tracing",
    "jit",
    "ops",
    "to_onnx",
    "to_string",
    "parser",
]
