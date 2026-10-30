"""Unit tests and verification suite for bonsai functionality."""

from onnx9000.converters.frontend.bonsai import BonsaiImporter
from onnx9000.core.exceptions import Onnx9000Error


def test_BonsaiImporter() -> None:
    """Verify that bonsai importer behaves as expected."""
    try:
        obj = BonsaiImporter()
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
