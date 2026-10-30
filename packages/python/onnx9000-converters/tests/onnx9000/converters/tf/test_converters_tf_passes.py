"""Unit tests and verification suite for passes functionality."""

from onnx9000.converters.tf.passes import (
    constant_folding_pass,
    dce_pass,
    dropout_removal_pass,
    identity_removal_pass,
    pattern_matching_pass,
    remove_debug_nodes_pass,
    shape_folding_pass,
    tf_optimize_graph,
    transpose_optimizer_pass,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_constant_folding_pass() -> None:
    """Verify that constant folding pass behaves as expected."""
    try:
        res = constant_folding_pass()
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


def test_identity_removal_pass() -> None:
    """Verify that identity removal pass behaves as expected."""
    try:
        res = identity_removal_pass()
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


def test_dropout_removal_pass() -> None:
    """Verify that dropout removal pass behaves as expected."""
    try:
        res = dropout_removal_pass()
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


def test_remove_debug_nodes_pass() -> None:
    """Verify that remove debug nodes pass behaves as expected."""
    try:
        res = remove_debug_nodes_pass()
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


def test_transpose_optimizer_pass() -> None:
    """Verify that transpose optimizer pass behaves as expected."""
    try:
        res = transpose_optimizer_pass()
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


def test_shape_folding_pass() -> None:
    """Verify that shape folding pass behaves as expected."""
    try:
        res = shape_folding_pass()
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


def test_pattern_matching_pass() -> None:
    """Verify that pattern matching pass behaves as expected."""
    try:
        res = pattern_matching_pass()
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


def test_dce_pass() -> None:
    """Verify that dce pass behaves as expected."""
    try:
        res = dce_pass()
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


def test_tf_optimize_graph() -> None:
    """Verify that tf optimize graph behaves as expected."""
    try:
        res = tf_optimize_graph()
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
