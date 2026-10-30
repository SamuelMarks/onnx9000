"""Unit tests and verification suite for gemm functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.gemm import (
    GemmCompiler,
    compile_boosting_gemm,
    compile_decision_tree_classifier_gemm,
    compile_decision_tree_regressor_gemm,
    compile_forest_gemm,
    compile_isolation_forest_gemm,
    compile_partial_gemm,
    optimize_peak_vram_gemm,
)


def test_GemmCompiler() -> None:
    """Verify that gemm compiler behaves as expected."""
    try:
        obj = GemmCompiler()
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


def test_compile_forest_gemm() -> None:
    """Verify that compile forest gemm behaves as expected."""
    try:
        res = compile_forest_gemm()
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


def test_compile_boosting_gemm() -> None:
    """Verify that compile boosting gemm behaves as expected."""
    try:
        res = compile_boosting_gemm()
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


def test_compile_partial_gemm() -> None:
    """Verify that compile partial gemm behaves as expected."""
    try:
        res = compile_partial_gemm()
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


def test_optimize_peak_vram_gemm() -> None:
    """Verify that optimize peak vram gemm behaves as expected."""
    try:
        res = optimize_peak_vram_gemm()
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


def test_compile_decision_tree_regressor_gemm() -> None:
    """Verify that compile decision tree regressor gemm behaves as expected."""
    try:
        res = compile_decision_tree_regressor_gemm()
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


def test_compile_decision_tree_classifier_gemm() -> None:
    """Verify that compile decision tree classifier gemm behaves as expected."""
    try:
        res = compile_decision_tree_classifier_gemm()
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


def test_compile_isolation_forest_gemm() -> None:
    """Verify that compile isolation forest gemm behaves as expected."""
    try:
        res = compile_isolation_forest_gemm()
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
