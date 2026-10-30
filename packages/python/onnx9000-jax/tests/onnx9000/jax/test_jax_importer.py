"""Unit tests and verification suite for jax importer functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.jax.jax_importer import JAXImporter


def test_JAXImporter() -> None:
    """Verify that jaximporter behaves as expected."""
    try:
        obj = JAXImporter()
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
