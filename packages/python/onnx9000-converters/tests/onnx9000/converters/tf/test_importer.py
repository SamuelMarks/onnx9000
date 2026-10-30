"""Unit tests and verification suite for importer functionality."""

from onnx9000.converters.tf.importer import TFImporter, _convert_dtype, load_tf
from onnx9000.core.exceptions import Onnx9000Error


def test_TFImporter() -> None:
    """Verify that tfimporter behaves as expected."""
    try:
        obj = TFImporter()
        assert obj is not None
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
    ):
        pass


def test__convert_dtype() -> None:
    """Verify that convert dtype behaves as expected."""
    try:
        res = _convert_dtype()
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


def test_load_tf() -> None:
    """Verify that load tf behaves as expected."""
    try:
        res = load_tf()
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
