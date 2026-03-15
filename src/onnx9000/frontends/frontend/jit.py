"""
Frontend Sub-Package

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from functools import wraps
from typing import Any, Callable

from onnx9000.frontends.frontend.builder import GraphBuilder, Tracing
from onnx9000.frontends.frontend.tensor import Tensor


def jit(fn: Callable) -> Callable:
    """
    Traces a Python function into an ONNX GraphBuilder.
    This is a symbolic tracer (similar to JAX's make_jaxpr or PyTorch's trace).
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Provides semantic functionality and verification."""
        # In a complete implementation, this would inspect args and dynamically
        # create symbolic Tensor inputs. For now, we assume users provide Tensors.

        """Provides wrapper functionality and verification."""

        builder = GraphBuilder(name=fn.__name__)

        # Track inputs
        for arg in args:
            if isinstance(arg, Tensor):
                builder.inputs.append(arg)

        # Handle parameters inside the function
        # A more sophisticated tracer would extract parameters from a Module class

        with Tracing(builder):
            outputs = fn(*args, **kwargs)

        # Standardize outputs to tuple/list
        if isinstance(outputs, tuple):
            builder.outputs.extend(outputs)
        elif isinstance(outputs, list):
            builder.outputs.extend(outputs)
        elif isinstance(outputs, Tensor):
            builder.outputs.append(outputs)

        return builder

    return wrapper
