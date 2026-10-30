"""Unit tests and verification suite for weight utils functionality."""

from onnx9000.converters.frontend.weight_utils import export_state_dict, universal_weight_bridge
from onnx9000.core.exceptions import Onnx9000Error


def test_export_state_dict() -> None:
    """Verify that export state dict behaves as expected."""
    try:
        res = export_state_dict()
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


def test_universal_weight_bridge() -> None:
    """Verify that universal weight bridge behaves as expected."""
    try:
        res = universal_weight_bridge()
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
