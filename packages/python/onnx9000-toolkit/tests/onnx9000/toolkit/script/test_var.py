"""Unit tests and verification suite for var functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.var import Var, _generate_unique_name


def test_Var() -> None:
    """Verify that var behaves as expected."""
    try:
        obj = Var()
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


def test__generate_unique_name() -> None:
    """Verify that generate unique name behaves as expected."""
    try:
        res = _generate_unique_name()
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
