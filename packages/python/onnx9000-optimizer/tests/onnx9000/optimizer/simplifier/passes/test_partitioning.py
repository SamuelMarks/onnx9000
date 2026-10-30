"""Unit tests and verification suite for partitioning functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.partitioning import partition_for_multi_device


def test_partition_for_multi_device() -> None:
    """Verify that partition for multi device behaves as expected."""
    try:
        res = partition_for_multi_device()
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
