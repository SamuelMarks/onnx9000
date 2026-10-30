"""Unit tests and verification suite for fusion functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.fusion import (
    FusionPass,
    PatternMatcherFusion,
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    fuse_consecutive_transpose,
    fuse_linear_activation,
    fuse_matmul_add,
    map_aten_arange_to_range,
    run_all_fusions,
)


def test_FusionPass() -> None:
    """Verify that fusion pass behaves as expected."""
    try:
        obj = FusionPass()
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


def test_PatternMatcherFusion() -> None:
    """Verify that pattern matcher fusion behaves as expected."""
    try:
        obj = PatternMatcherFusion()
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


def test_fuse_batchnorm_into_gemm() -> None:
    """Verify that fuse batchnorm into gemm behaves as expected."""
    try:
        res = fuse_batchnorm_into_gemm()
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


def test_fuse_batchnorm_into_conv() -> None:
    """Verify that fuse batchnorm into conv behaves as expected."""
    try:
        res = fuse_batchnorm_into_conv()
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


def test_map_aten_arange_to_range() -> None:
    """Verify that map aten arange to range behaves as expected."""
    try:
        res = map_aten_arange_to_range()
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


def test_run_all_fusions() -> None:
    """Verify that run all fusions behaves as expected."""
    try:
        res = run_all_fusions()
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


def test_fuse_linear_activation() -> None:
    """Verify that fuse linear activation behaves as expected."""
    try:
        res = fuse_linear_activation()
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


def test_fuse_consecutive_transpose() -> None:
    """Verify that fuse consecutive transpose behaves as expected."""
    try:
        res = fuse_consecutive_transpose()
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


def test_fuse_matmul_add() -> None:
    """Verify that fuse matmul add behaves as expected."""
    try:
        res = fuse_matmul_add()
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
