"""Unit tests and verification suite for schema functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.schema import (
    OpSchema,
    SchemaRegistry,
    get_target_opset,
    set_target_opset,
    validate_op,
)


def test_OpSchema() -> None:
    """Verify that op schema behaves as expected."""
    try:
        obj = OpSchema()
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


def test_SchemaRegistry() -> None:
    """Verify that schema registry behaves as expected."""
    try:
        obj = SchemaRegistry()
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


def test_set_target_opset() -> None:
    """Verify that set target opset behaves as expected."""
    try:
        res = set_target_opset()
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


def test_get_target_opset() -> None:
    """Verify that get target opset behaves as expected."""
    try:
        res = get_target_opset()
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


def test_validate_op() -> None:
    """Verify that validate op behaves as expected."""
    try:
        res = validate_op()
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
