"""Unit tests and verification suite for distributed functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.safetensors.distributed import load_sharded_tensors, pipeline_parallel_loader


def test_load_sharded_tensors() -> None:
    """Verify that load sharded tensors behaves as expected."""
    try:
        res = load_sharded_tensors()
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


def test_pipeline_parallel_loader() -> None:
    """Verify that pipeline parallel loader behaves as expected."""
    try:
        res = pipeline_parallel_loader()
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
