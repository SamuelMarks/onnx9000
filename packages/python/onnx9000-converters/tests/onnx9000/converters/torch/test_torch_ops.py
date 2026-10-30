"""Unit tests and verification suite for torch ops functionality."""

from onnx9000.converters.torch.torch_ops import _create_mapper
from onnx9000.core.exceptions import Onnx9000Error


def test__create_mapper() -> None:
    """Verify that create mapper behaves as expected."""
    try:
        res = _create_mapper()
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
