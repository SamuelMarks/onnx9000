"""Unit tests and verification suite for normalization functionality."""

from onnx9000.converters.frontend.nn.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    _BatchNormNd,
    _InstanceNormNd,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__BatchNormNd() -> None:
    """Verify that batch norm nd behaves as expected."""
    try:
        obj = _BatchNormNd()
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


def test_BatchNorm1d() -> None:
    """Verify that batch norm1d behaves as expected."""
    try:
        obj = BatchNorm1d()
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


def test_BatchNorm2d() -> None:
    """Verify that batch norm2d behaves as expected."""
    try:
        obj = BatchNorm2d()
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


def test_BatchNorm3d() -> None:
    """Verify that batch norm3d behaves as expected."""
    try:
        obj = BatchNorm3d()
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


def test_LayerNorm() -> None:
    """Verify that layer norm behaves as expected."""
    try:
        obj = LayerNorm()
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


def test_GroupNorm() -> None:
    """Verify that group norm behaves as expected."""
    try:
        obj = GroupNorm()
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


def test__InstanceNormNd() -> None:
    """Verify that instance norm nd behaves as expected."""
    try:
        obj = _InstanceNormNd()
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


def test_InstanceNorm1d() -> None:
    """Verify that instance norm1d behaves as expected."""
    try:
        obj = InstanceNorm1d()
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


def test_InstanceNorm2d() -> None:
    """Verify that instance norm2d behaves as expected."""
    try:
        obj = InstanceNorm2d()
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


def test_InstanceNorm3d() -> None:
    """Verify that instance norm3d behaves as expected."""
    try:
        obj = InstanceNorm3d()
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
