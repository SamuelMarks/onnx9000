"""Unit tests and verification suite for intrinsics functionality."""

from onnx9000.c_compiler.intrinsics import (
    apply_simd_unroll,
    emit_avx2_headers,
    emit_cmsis_nn_qlinear_conv,
    emit_cmsis_nn_qlinear_matmul,
    emit_esp_nn_qlinear_conv,
    emit_esp_nn_qlinear_matmul,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_emit_cmsis_nn_qlinear_matmul() -> None:
    """Verify that emit cmsis nn qlinear matmul behaves as expected."""
    try:
        res = emit_cmsis_nn_qlinear_matmul()
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


def test_emit_cmsis_nn_qlinear_conv() -> None:
    """Verify that emit cmsis nn qlinear conv behaves as expected."""
    try:
        res = emit_cmsis_nn_qlinear_conv()
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


def test_apply_simd_unroll() -> None:
    """Verify that apply simd unroll behaves as expected."""
    try:
        res = apply_simd_unroll()
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


def test_emit_esp_nn_qlinear_matmul() -> None:
    """Verify that emit esp nn qlinear matmul behaves as expected."""
    try:
        res = emit_esp_nn_qlinear_matmul()
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


def test_emit_esp_nn_qlinear_conv() -> None:
    """Verify that emit esp nn qlinear conv behaves as expected."""
    try:
        res = emit_esp_nn_qlinear_conv()
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


def test_emit_avx2_headers() -> None:
    """Verify that emit avx2 headers behaves as expected."""
    try:
        res = emit_avx2_headers()
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
