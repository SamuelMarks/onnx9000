"""Frontend Sub-Package.

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from collections.abc import Callable
from functools import wraps

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.tensor import Tensor


def jit(fn: Callable) -> Callable:
    """Traces a Python function into an ONNX GraphBuilder.

    This is a symbolic tracer (similar to JAX's make_jaxpr or PyTorch's trace).
    """

    @wraps(fn)
    def wrapper(
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute wrapper operation.

        Args:
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of wrapper (Union[int, float, str, bool, bytes, list, dict, tuple, None]).
        """
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
