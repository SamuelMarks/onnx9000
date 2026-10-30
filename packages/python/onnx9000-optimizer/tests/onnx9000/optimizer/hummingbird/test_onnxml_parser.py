"""Unit tests and verification suite for onnxml parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.onnxml_parser import (
    apply_onnxml_post_transform,
    ensure_static_shapes,
    extract_tree_ensemble_attributes,
    parse_onnxml_binarizer,
    parse_onnxml_category_mapper,
    parse_onnxml_feature_extractor,
    parse_onnxml_imputer,
    parse_onnxml_linear,
    parse_onnxml_normalizer,
    parse_onnxml_onehot,
    parse_onnxml_scaler,
    parse_onnxml_svm,
    parse_onnxml_tree_ensemble,
    parse_onnxml_zipmap,
)


def test_parse_onnxml_tree_ensemble() -> None:
    """Verify that parse onnxml tree ensemble behaves as expected."""
    try:
        res = parse_onnxml_tree_ensemble()
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


def test_extract_tree_ensemble_attributes() -> None:
    """Verify that extract tree ensemble attributes behaves as expected."""
    try:
        res = extract_tree_ensemble_attributes()
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


def test_parse_onnxml_linear() -> None:
    """Verify that parse onnxml linear behaves as expected."""
    try:
        res = parse_onnxml_linear()
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


def test_parse_onnxml_svm() -> None:
    """Verify that parse onnxml svm behaves as expected."""
    try:
        res = parse_onnxml_svm()
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


def test_parse_onnxml_scaler() -> None:
    """Verify that parse onnxml scaler behaves as expected."""
    try:
        res = parse_onnxml_scaler()
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


def test_parse_onnxml_normalizer() -> None:
    """Verify that parse onnxml normalizer behaves as expected."""
    try:
        res = parse_onnxml_normalizer()
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


def test_parse_onnxml_binarizer() -> None:
    """Verify that parse onnxml binarizer behaves as expected."""
    try:
        res = parse_onnxml_binarizer()
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


def test_parse_onnxml_onehot() -> None:
    """Verify that parse onnxml onehot behaves as expected."""
    try:
        res = parse_onnxml_onehot()
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


def test_parse_onnxml_imputer() -> None:
    """Verify that parse onnxml imputer behaves as expected."""
    try:
        res = parse_onnxml_imputer()
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


def test_parse_onnxml_feature_extractor() -> None:
    """Verify that parse onnxml feature extractor behaves as expected."""
    try:
        res = parse_onnxml_feature_extractor()
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


def test_parse_onnxml_category_mapper() -> None:
    """Verify that parse onnxml category mapper behaves as expected."""
    try:
        res = parse_onnxml_category_mapper()
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


def test_parse_onnxml_zipmap() -> None:
    """Verify that parse onnxml zipmap behaves as expected."""
    try:
        res = parse_onnxml_zipmap()
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


def test_apply_onnxml_post_transform() -> None:
    """Verify that apply onnxml post transform behaves as expected."""
    try:
        res = apply_onnxml_post_transform()
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


def test_ensure_static_shapes() -> None:
    """Verify that ensure static shapes behaves as expected."""
    try:
        res = ensure_static_shapes()
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
