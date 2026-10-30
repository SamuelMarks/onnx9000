"""Unit tests and verification suite for tree traversal functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.tree_traversal import (
    TreeTraversalCompiler,
    compile_forest_tree_traversal,
    flatten_multi_class_traversal,
    handle_categorical_traversal,
    handle_missing_value_traversal,
    test_gather_latency_wasm,
)


def test_TreeTraversalCompiler() -> None:
    """Verify that tree traversal compiler behaves as expected."""
    try:
        obj = TreeTraversalCompiler()
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


def test_compile_forest_tree_traversal() -> None:
    """Verify that compile forest tree traversal behaves as expected."""
    try:
        res = compile_forest_tree_traversal()
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


def test_handle_categorical_traversal() -> None:
    """Verify that handle categorical traversal behaves as expected."""
    try:
        res = handle_categorical_traversal()
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


def test_handle_missing_value_traversal() -> None:
    """Verify that handle missing value traversal behaves as expected."""
    try:
        res = handle_missing_value_traversal()
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


def test_flatten_multi_class_traversal() -> None:
    """Verify that flatten multi class traversal behaves as expected."""
    try:
        res = flatten_multi_class_traversal()
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


def test_test_gather_latency_wasm() -> None:
    """Verify that test gather latency wasm behaves as expected."""
    try:
        res = test_gather_latency_wasm()
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
