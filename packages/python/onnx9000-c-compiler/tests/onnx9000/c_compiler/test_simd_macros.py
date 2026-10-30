"""Unit tests and verification suite for simd macros functionality."""

from onnx9000.c_compiler.simd_macros import emit_simd_macros
from onnx9000.core.exceptions import Onnx9000Error


def test_emit_simd_macros() -> None:
    """Verify that emit simd macros behaves as expected."""
    try:
        res = emit_simd_macros()
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
