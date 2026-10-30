"""Tracer framework."""

import threading
from collections.abc import Callable

from onnx9000.converters.frontend.builder import GraphBuilder, get_active_builder
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.exceptions import ConversionError, Onnx9000Error

_tls = threading.local()


class Tracer:
    """Context manager for tracing operations on Tensor objects to build an ONNX graph.

    This class intercepts operations on Tensor objects by setting an active GraphBuilder
    in a thread-local storage.
    """

    def __init__(self, builder: GraphBuilder | None = None) -> None:
        """Initialize the Tracer with an optional GraphBuilder.

        Args:
            builder: The GraphBuilder to use for recording operations. If None, a new one is created.

        """
        self.builder = builder or GraphBuilder()
        self.prev_builder: GraphBuilder | None = None

    def __enter__(self) -> GraphBuilder:
        """Activate the tracer, making its builder the active one in thread-local storage.

        Returns:
            The GraphBuilder instance associated with this tracer.

        """
        self.prev_builder = get_active_builder()
        import onnx9000.converters.frontend.builder as builder_mod

        builder_mod._tls.builder = self.builder
        _tls.tracer = self
        return self.builder

    def __exit__(
        self,
        exc_type: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_val: int | float | str | bool | bytes | list | dict | tuple | None,
        exc_tb: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Deactivate the tracer and restore the previous GraphBuilder."""
        import onnx9000.converters.frontend.builder as builder_mod

        builder_mod._tls.builder = self.prev_builder
        _tls.tracer = getattr(_tls, "prev_tracer", None)


class Proxy(Tensor):
    """Proxy tensor object that records operations instead of executing them."""

    def __init__(self, name: str) -> None:
        """Initialize Proxy."""
        from onnx9000.core.dtypes import DType

        super().__init__((1,), DType.FLOAT32, name=name)


def trace(
    func: int | float | str | bool | bytes | list | dict | tuple | None,
    *args: int | float | str | bool | bytes | list | dict | tuple | None,
    **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
) -> GraphBuilder:
    """Traces a Python function into an ONNX GraphBuilder."""
    from onnx9000.converters.frontend.tree import find_tensors, tree_map

    name = getattr(func, "__name__", func.__class__.__name__)
    builder = GraphBuilder(name=name)

    def make_proxy(
        arg: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Recursively convert Tensor objects in arguments to proxies for tracing."""
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
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ) as e:
        raise ConversionError(f"Tracing failed: {e}") from e

    if outputs is not None:
        builder.outputs.extend(find_tensors(outputs))

    return builder


def script(
    func: Callable,
    *args: int | float | str | bool | bytes | list | dict | tuple | None,
    **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
) -> GraphBuilder:
    """Script a Python function using TorchScript IR if available, otherwise AST translation."""
    try:
        import torch

        _ = torch
        from onnx9000.converters.torch.script import TorchScriptParser

        parser = TorchScriptParser(func)
        return parser.parse()
    except (ImportError, Exception):
        from onnx9000.converters.frontend.ast_parser import ScriptCompiler

        compiler = ScriptCompiler(func)
        return compiler.compile(*args, **kwargs)
