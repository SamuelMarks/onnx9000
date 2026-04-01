"""Tracer framework."""

import threading
from typing import Any, Callable, Optional

from onnx9000.converters.frontend.builder import GraphBuilder, get_active_builder
from onnx9000.converters.frontend.tensor import Tensor

_tls = threading.local()


class Tracer:
    """Context manager that intercepts all Tensor operations."""

    def __init__(self, builder: Optional[GraphBuilder] = None) -> None:
        """Implement the __init__ method."""
        self.builder = builder or GraphBuilder()
        self.prev_builder: Optional[GraphBuilder] = None

    def __enter__(self) -> GraphBuilder:
        """Implement the __enter__ method."""
        self.prev_builder = get_active_builder()
        import onnx9000.converters.frontend.builder as builder_mod

        builder_mod._tls.builder = self.builder
        _tls.tracer = self
        return self.builder

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Implement the __exit__ method."""
        import onnx9000.converters.frontend.builder as builder_mod

        builder_mod._tls.builder = self.prev_builder
        _tls.tracer = getattr(_tls, "prev_tracer", None)


class Proxy(Tensor):
    """Proxy tensor object that records operations instead of executing them."""


def trace(func: Any, *args: Any, **kwargs: Any) -> GraphBuilder:
    """Traces a Python function into an ONNX GraphBuilder."""
    from onnx9000.converters.frontend.tree import find_tensors, tree_map

    name = getattr(func, "__name__", func.__class__.__name__)
    builder = GraphBuilder(name=name)

    def make_proxy(arg: Any) -> Any:
        if isinstance(arg, Tensor):
            p = arg.__class__(arg.shape, arg.dtype, name=arg.name)
            p.data = getattr(arg, "data", None)
            if not any(p.name == i.name for i in builder.inputs):
                builder.inputs.append(p)
            return p
        return arg

    proxy_args = [tree_map(make_proxy, arg) for arg in args]
    proxy_kwargs = {k: tree_map(make_proxy, v) for k, v in kwargs.items()}

    try:
        with Tracer(builder):
            outputs = func(*proxy_args, **proxy_kwargs)
    except Exception as e:
        raise RuntimeError(f"Tracing failed: {e}") from e

    if outputs is not None:
        builder.outputs.extend(find_tensors(outputs))

    return builder


def script(func: Callable, *args: Any, **kwargs: Any) -> GraphBuilder:
    """Script a Python function using TorchScript IR if available, otherwise AST translation."""
    try:
        import torch
        from onnx9000.converters.torch.script import TorchScriptParser

        parser = TorchScriptParser(func)
        return parser.parse()
    except (ImportError, Exception):
        from onnx9000.converters.frontend.ast_parser import ScriptCompiler

        compiler = ScriptCompiler(func)
        return compiler.compile(*args, **kwargs)
