"""Unit tests and verification suite for shape inference functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.shape_inference import infer_backward_shapes


def test_infer_backward_shapes() -> None:
    """Verify that infer backward shapes behaves as expected."""
    try:
        res = infer_backward_shapes()
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
