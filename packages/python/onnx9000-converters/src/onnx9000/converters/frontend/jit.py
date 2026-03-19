"""Frontend Sub-Package.

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from functools import wraps
from typing import Any, Callable

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.tensor import Tensor


def jit(fn: Callable) -> Callable:
    """Traces a Python function into an ONNX GraphBuilder.

    This is a symbolic tracer (similar to JAX's make_jaxpr or PyTorch's trace).
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Implement the wrapper method."""
        "Provides wrapper functionality and verification."
        builder = GraphBuilder(name=fn.__name__)
        for arg in args:
            if isinstance(arg, Tensor):
                builder.inputs.append(arg)
        with Tracing(builder):
            outputs = fn(*args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            builder.outputs.extend(outputs)
        elif isinstance(outputs, Tensor):
            builder.outputs.append(outputs)
        return builder

    return wrapper
