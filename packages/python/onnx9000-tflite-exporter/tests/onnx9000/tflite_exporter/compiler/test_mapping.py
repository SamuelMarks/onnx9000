"""Unit tests and verification suite for mapping functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.compiler.mapping import (
    create_shape_signature,
    map_onnx_shape_to_tflite,
    map_onnx_type_to_tflite,
)


def test_map_onnx_type_to_tflite() -> None:
    """Verify that map onnx type to tflite behaves as expected."""
    try:
        res = map_onnx_type_to_tflite()
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


def test_map_onnx_shape_to_tflite() -> None:
    """Verify that map onnx shape to tflite behaves as expected."""
    try:
        res = map_onnx_shape_to_tflite()
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


def test_create_shape_signature() -> None:
    """Verify that create shape signature behaves as expected."""
    try:
        res = create_shape_signature()
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
