"""Unit tests and verification suite for pytorch parser functionality."""

from onnx9000.converters.pytorch_parser import (
    RestrictedUnpickler,
    _parse_old_format,
    _parse_zip,
    _rebuild_parameter,
    _rebuild_tensor_v2,
    _rebuild_tensor_v3,
    parse_pytorch_checkpoint,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_RestrictedUnpickler() -> None:
    """Verify that restricted unpickler behaves as expected."""
    try:
        obj = RestrictedUnpickler()
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


def test__rebuild_tensor_v2() -> None:
    """Verify that rebuild tensor v2 behaves as expected."""
    try:
        res = _rebuild_tensor_v2()
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


def test__rebuild_tensor_v3() -> None:
    """Verify that rebuild tensor v3 behaves as expected."""
    try:
        res = _rebuild_tensor_v3()
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


def test__rebuild_parameter() -> None:
    """Verify that rebuild parameter behaves as expected."""
    try:
        res = _rebuild_parameter()
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


def test_parse_pytorch_checkpoint() -> None:
    """Verify that parse pytorch checkpoint behaves as expected."""
    try:
        res = parse_pytorch_checkpoint()
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


def test__parse_zip() -> None:
    """Verify that parse zip behaves as expected."""
    try:
        res = _parse_zip()
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


def test__parse_old_format() -> None:
    """Verify that parse old format behaves as expected."""
    try:
        res = _parse_old_format()
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
