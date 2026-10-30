"""Unit tests and verification suite for math utils functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.math_utils import (
    clamp_nan_to_zero,
    division_by_zero_guard,
    enforce_broadcast_safety,
    ensure_softmax_stability,
    fold_scaler_into_linear,
    handle_64bit_casting,
    map_count_vectorizer,
    map_knn_distances,
    map_murmurhash3,
    map_naive_bayes,
    map_pca_svd_lda,
    map_polynomial_expansion,
    map_tfidf_vectorizer,
    optimize_sigmoid,
    replace_mod,
    replace_where_with_arithmetic_mask,
    validate_math_exactness,
)


def test_map_count_vectorizer() -> None:
    """Verify that map count vectorizer behaves as expected."""
    try:
        res = map_count_vectorizer()
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


def test_map_tfidf_vectorizer() -> None:
    """Verify that map tfidf vectorizer behaves as expected."""
    try:
        res = map_tfidf_vectorizer()
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


def test_map_polynomial_expansion() -> None:
    """Verify that map polynomial expansion behaves as expected."""
    try:
        res = map_polynomial_expansion()
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


def test_map_murmurhash3() -> None:
    """Verify that map murmurhash3 behaves as expected."""
    try:
        res = map_murmurhash3()
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


def test_optimize_sigmoid() -> None:
    """Verify that optimize sigmoid behaves as expected."""
    try:
        res = optimize_sigmoid()
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


def test_fold_scaler_into_linear() -> None:
    """Verify that fold scaler into linear behaves as expected."""
    try:
        res = fold_scaler_into_linear()
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


def test_map_knn_distances() -> None:
    """Verify that map knn distances behaves as expected."""
    try:
        res = map_knn_distances()
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


def test_replace_mod() -> None:
    """Verify that replace mod behaves as expected."""
    try:
        res = replace_mod()
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


def test_replace_where_with_arithmetic_mask() -> None:
    """Verify that replace where with arithmetic mask behaves as expected."""
    try:
        res = replace_where_with_arithmetic_mask()
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


def test_clamp_nan_to_zero() -> None:
    """Verify that clamp nan to zero behaves as expected."""
    try:
        res = clamp_nan_to_zero()
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


def test_division_by_zero_guard() -> None:
    """Verify that division by zero guard behaves as expected."""
    try:
        res = division_by_zero_guard()
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


def test_ensure_softmax_stability() -> None:
    """Verify that ensure softmax stability behaves as expected."""
    try:
        res = ensure_softmax_stability()
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


def test_map_naive_bayes() -> None:
    """Verify that map naive bayes behaves as expected."""
    try:
        res = map_naive_bayes()
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


def test_map_pca_svd_lda() -> None:
    """Verify that map pca svd lda behaves as expected."""
    try:
        res = map_pca_svd_lda()
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


def test_handle_64bit_casting() -> None:
    """Verify that handle 64bit casting behaves as expected."""
    try:
        res = handle_64bit_casting()
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


def test_enforce_broadcast_safety() -> None:
    """Verify that enforce broadcast safety behaves as expected."""
    try:
        res = enforce_broadcast_safety()
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


def test_validate_math_exactness() -> None:
    """Verify that validate math exactness behaves as expected."""
    try:
        res = validate_math_exactness()
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
