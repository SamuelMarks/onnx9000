"""Tracer framework."""

from typing import Any, Optional, Callable, Dict, Tuple
from onnx9000.frontends.frontend.builder import GraphBuilder, get_active_builder
from onnx9000.frontends.frontend.tensor import Tensor
import threading

_tls = threading.local()


class Tracer:
    """Context manager that intercepts all Tensor operations."""

    def __init__(self, builder: Optional[GraphBuilder] = None) -> None:
        """Provides semantic functionality and verification."""
        self.builder = builder or GraphBuilder()
        self.prev_builder: Optional[GraphBuilder] = None

    def __enter__(self) -> GraphBuilder:
        """Provides semantic functionality and verification."""
        self.prev_builder = get_active_builder()
        import onnx9000.frontends.frontend.builder as builder_mod

        builder_mod._tls.builder = self.builder
        _tls.tracer = self
        return self.builder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Provides semantic functionality and verification."""
        import onnx9000.frontends.frontend.builder as builder_mod

        builder_mod._tls.builder = self.prev_builder
        _tls.tracer = getattr(_tls, "prev_tracer", None)


class Proxy(Tensor):
    """Proxy tensor object that records operations instead of executing them."""


def trace(func: Any, *args: Any, **kwargs: Any) -> GraphBuilder:
    """Traces a Python function into an ONNX GraphBuilder."""
    name = getattr(func, "__name__", func.__class__.__name__)
    builder = GraphBuilder(name=name)
    proxy_args = []
    for i, arg in enumerate(args):
        if isinstance(arg, Tensor):
            name = getattr(arg, "name", None) or f"input_{i}"
            p = arg.__class__(arg.shape, arg.dtype, name=name)
            p.data = getattr(arg, "data", None)
            builder.inputs.append(p)
            proxy_args.append(p)
        else:
            proxy_args.append(arg)
    proxy_kwargs = {}
    for k, arg in kwargs.items():
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            name = getattr(arg, "name", None) or f"kwinput_{k}"
            p = arg.__class__(arg.shape, arg.dtype, name=name)
            p.data = getattr(arg, "data", None)
            builder.inputs.append(p)
            proxy_kwargs[k] = p
        else:
            proxy_kwargs[k] = arg
    try:
        with Tracer(builder):
            outputs = func(*proxy_args, **proxy_kwargs)
    except Exception as e:
        raise RuntimeError(f"Tracing failed: {e}") from e

    if outputs is not None:
        if isinstance(outputs, tuple):
            builder.outputs.extend(outputs)

        elif isinstance(outputs, dict):
            builder.outputs.extend(outputs.values())
        else:
            builder.outputs.append(outputs)

    return builder


def script(func: Callable, *args: Any, **kwargs: Any) -> GraphBuilder:
    """Script a Python function using AST translation."""
    from onnx9000.frontends.frontend.ast_parser import ScriptCompiler

    compiler = ScriptCompiler(func)
    return compiler.compile(*args, **kwargs)
