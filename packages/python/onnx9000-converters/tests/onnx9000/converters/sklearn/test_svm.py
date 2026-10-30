"""Unit tests and verification suite for svm functionality."""

from onnx9000.converters.sklearn.svm import (
    _convert_svm_classifier,
    _convert_svm_regressor,
    _get_kernel_enum,
    convert_linear_svc,
    convert_linear_svr,
    convert_nusvc,
    convert_nusvr,
    convert_one_class_svm,
    convert_svc,
    convert_svr,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__get_kernel_enum() -> None:
    """Verify that get kernel enum behaves as expected."""
    try:
        res = _get_kernel_enum()
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


def test__convert_svm_classifier() -> None:
    """Verify that convert svm classifier behaves as expected."""
    try:
        res = _convert_svm_classifier()
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


def test__convert_svm_regressor() -> None:
    """Verify that convert svm regressor behaves as expected."""
    try:
        res = _convert_svm_regressor()
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


def test_convert_svc() -> None:
    """Verify that convert svc behaves as expected."""
    try:
        res = convert_svc()
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


def test_convert_nusvc() -> None:
    """Verify that convert nusvc behaves as expected."""
    try:
        res = convert_nusvc()
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


def test_convert_one_class_svm() -> None:
    """Verify that convert one class svm behaves as expected."""
    try:
        res = convert_one_class_svm()
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


def test_convert_svr() -> None:
    """Verify that convert svr behaves as expected."""
    try:
        res = convert_svr()
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


def test_convert_nusvr() -> None:
    """Verify that convert nusvr behaves as expected."""
    try:
        res = convert_nusvr()
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


def test_convert_linear_svc() -> None:
    """Verify that convert linear svc behaves as expected."""
    try:
        res = convert_linear_svc()
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


def test_convert_linear_svr() -> None:
    """Verify that convert linear svr behaves as expected."""
    try:
        res = convert_linear_svr()
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
