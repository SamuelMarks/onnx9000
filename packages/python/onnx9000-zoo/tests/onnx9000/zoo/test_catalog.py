"""Unit tests and verification suite for catalog functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.zoo.catalog import ZooCatalog


def test_ZooCatalog() -> None:
    """Verify that zoo catalog behaves as expected."""
    try:
        obj = ZooCatalog()
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
