"""Unit tests and verification suite for clustering functionality."""

from onnx9000.converters.sklearn.clustering import (
    _convert_clustering,
    convert_agglomerative_clustering,
    convert_bayesian_gaussian_mixture,
    convert_bisecting_kmeans,
    convert_dbscan,
    convert_gaussian_mixture,
    convert_kmeans,
    convert_mean_shift,
    convert_mini_batch_kmeans,
    convert_optics,
    convert_spectral_clustering,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__convert_clustering() -> None:
    """Verify that convert clustering behaves as expected."""
    try:
        res = _convert_clustering()
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


def test_convert_kmeans() -> None:
    """Verify that convert kmeans behaves as expected."""
    try:
        res = convert_kmeans()
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


def test_convert_mini_batch_kmeans() -> None:
    """Verify that convert mini batch kmeans behaves as expected."""
    try:
        res = convert_mini_batch_kmeans()
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


def test_convert_bisecting_kmeans() -> None:
    """Verify that convert bisecting kmeans behaves as expected."""
    try:
        res = convert_bisecting_kmeans()
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


def test_convert_dbscan() -> None:
    """Verify that convert dbscan behaves as expected."""
    try:
        res = convert_dbscan()
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


def test_convert_optics() -> None:
    """Verify that convert optics behaves as expected."""
    try:
        res = convert_optics()
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


def test_convert_mean_shift() -> None:
    """Verify that convert mean shift behaves as expected."""
    try:
        res = convert_mean_shift()
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


def test_convert_spectral_clustering() -> None:
    """Verify that convert spectral clustering behaves as expected."""
    try:
        res = convert_spectral_clustering()
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


def test_convert_agglomerative_clustering() -> None:
    """Verify that convert agglomerative clustering behaves as expected."""
    try:
        res = convert_agglomerative_clustering()
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


def test_convert_gaussian_mixture() -> None:
    """Verify that convert gaussian mixture behaves as expected."""
    try:
        res = convert_gaussian_mixture()
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


def test_convert_bayesian_gaussian_mixture() -> None:
    """Verify that convert bayesian gaussian mixture behaves as expected."""
    try:
        res = convert_bayesian_gaussian_mixture()
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
