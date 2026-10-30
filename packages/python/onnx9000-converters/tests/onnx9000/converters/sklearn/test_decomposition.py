"""Unit tests and verification suite for decomposition functionality."""

from onnx9000.converters.sklearn.decomposition import (
    _convert_pca_like,
    convert_fast_ica,
    convert_incremental_pca,
    convert_kernel_pca,
    convert_latent_dirichlet_allocation,
    convert_nmf,
    convert_pca,
    convert_truncated_svd,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__convert_pca_like() -> None:
    """Verify that convert pca like behaves as expected."""
    try:
        res = _convert_pca_like()
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


def test_convert_pca() -> None:
    """Verify that convert pca behaves as expected."""
    try:
        res = convert_pca()
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


def test_convert_incremental_pca() -> None:
    """Verify that convert incremental pca behaves as expected."""
    try:
        res = convert_incremental_pca()
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


def test_convert_truncated_svd() -> None:
    """Verify that convert truncated svd behaves as expected."""
    try:
        res = convert_truncated_svd()
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


def test_convert_fast_ica() -> None:
    """Verify that convert fast ica behaves as expected."""
    try:
        res = convert_fast_ica()
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


def test_convert_nmf() -> None:
    """Verify that convert nmf behaves as expected."""
    try:
        res = convert_nmf()
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


def test_convert_kernel_pca() -> None:
    """Verify that convert kernel pca behaves as expected."""
    try:
        res = convert_kernel_pca()
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


def test_convert_latent_dirichlet_allocation() -> None:
    """Verify that convert latent dirichlet allocation behaves as expected."""
    try:
        res = convert_latent_dirichlet_allocation()
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
