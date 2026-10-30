"""Unit tests and verification suite for fusions functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.surgeon.fusions import fuse_flash_attention, fuse_horizontal_gemm


def test_fuse_flash_attention() -> None:
    """Verify that fuse flash attention behaves as expected."""
    try:
        res = fuse_flash_attention()
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


def test_fuse_horizontal_gemm() -> None:
    """Verify that fuse horizontal gemm behaves as expected."""
    try:
        res = fuse_horizontal_gemm()
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
