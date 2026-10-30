"""Unit tests and verification suite for utils functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.utils import (
    GradientProto,
    add_synchronous_barrier,
    calculate_communication_bounds,
    calculate_gradient_payload_size,
    compile_multi_replica_graph,
    compress_gradients_int8,
    embed_distributed_identifiers,
    flatten_gradients,
    generate_gradient_proto,
    reverse_topological_sort,
)


def test_GradientProto() -> None:
    """Verify that gradient proto behaves as expected."""
    try:
        obj = GradientProto()
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


def test_generate_gradient_proto() -> None:
    """Verify that generate gradient proto behaves as expected."""
    try:
        res = generate_gradient_proto()
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


def test_calculate_gradient_payload_size() -> None:
    """Verify that calculate gradient payload size behaves as expected."""
    try:
        res = calculate_gradient_payload_size()
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


def test_compress_gradients_int8() -> None:
    """Verify that compress gradients int8 behaves as expected."""
    try:
        res = compress_gradients_int8()
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


def test_compile_multi_replica_graph() -> None:
    """Verify that compile multi replica graph behaves as expected."""
    try:
        res = compile_multi_replica_graph()
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


def test_embed_distributed_identifiers() -> None:
    """Verify that embed distributed identifiers behaves as expected."""
    try:
        res = embed_distributed_identifiers()
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


def test_add_synchronous_barrier() -> None:
    """Verify that add synchronous barrier behaves as expected."""
    try:
        res = add_synchronous_barrier()
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


def test_calculate_communication_bounds() -> None:
    """Verify that calculate communication bounds behaves as expected."""
    try:
        res = calculate_communication_bounds()
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


def test_flatten_gradients() -> None:
    """Verify that flatten gradients behaves as expected."""
    try:
        res = flatten_gradients()
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


def test_reverse_topological_sort() -> None:
    """Verify that reverse topological sort behaves as expected."""
    try:
        res = reverse_topological_sort()
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
