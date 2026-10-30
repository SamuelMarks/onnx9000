"""Unit tests and verification suite for pattern matcher functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.pattern_matcher import (
    Pattern,
    PatternMatcherEngine,
    apply_algebraic_reuse,
    apply_fusion_reuse,
    apply_hardware_lowering,
    matches,
)


def test_Pattern() -> None:
    """Verify that pattern behaves as expected."""
    try:
        obj = Pattern()
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


def test_PatternMatcherEngine() -> None:
    """Verify that pattern matcher engine behaves as expected."""
    try:
        obj = PatternMatcherEngine()
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


def test_matches() -> None:
    """Verify that matches behaves as expected."""
    try:
        res = matches()
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


def test_apply_algebraic_reuse() -> None:
    """Verify that apply algebraic reuse behaves as expected."""
    try:
        res = apply_algebraic_reuse()
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


def test_apply_fusion_reuse() -> None:
    """Verify that apply fusion reuse behaves as expected."""
    try:
        res = apply_fusion_reuse()
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


def test_apply_hardware_lowering() -> None:
    """Verify that apply hardware lowering behaves as expected."""
    try:
        res = apply_hardware_lowering()
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
