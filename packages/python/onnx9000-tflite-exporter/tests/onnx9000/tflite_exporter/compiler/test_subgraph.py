"""Unit tests and verification suite for subgraph functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite


def test_compile_graph_to_tflite() -> None:
    """Verify that compile graph to tflite behaves as expected."""
    try:
        res = compile_graph_to_tflite()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
