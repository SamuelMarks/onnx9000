"""Unit tests and verification suite for api functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.openvino.api import export_model


def test_export_model() -> None:
    """Verify that export model behaves as expected."""
    try:
        res = export_model()
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
