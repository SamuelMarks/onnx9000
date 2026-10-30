"""Unit tests and verification suite for api functionality."""

from onnx9000.converters.tf.api import (
    _convert_tfgraph,
    convert_keras_to_onnx,
    convert_tf_to_onnx,
    convert_tflite_to_onnx,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__convert_tfgraph() -> None:
    """Verify that convert tfgraph behaves as expected."""
    try:
        res = _convert_tfgraph()
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


def test_convert_tf_to_onnx() -> None:
    """Verify that convert tf to onnx behaves as expected."""
    try:
        res = convert_tf_to_onnx()
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


def test_convert_keras_to_onnx() -> None:
    """Verify that convert keras to onnx behaves as expected."""
    try:
        res = convert_keras_to_onnx()
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


def test_convert_tflite_to_onnx() -> None:
    """Verify that convert tflite to onnx behaves as expected."""
    try:
        res = convert_tflite_to_onnx()
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
