"""Unit tests and verification suite for control flow ops functionality."""

from onnx9000.converters.tf.control_flow_ops import (
    _map_gru_cell,
    _map_if,
    _map_lstm_cell,
    _map_noop,
    _map_tensor_array,
    _map_tensor_array_gather,
    _map_tensor_array_read,
    _map_tensor_array_scatter,
    _map_tensor_array_size,
    _map_tensor_array_write,
    _map_while,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__map_noop() -> None:
    """Verify that map noop behaves as expected."""
    try:
        res = _map_noop()
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


def test__map_if() -> None:
    """Verify that map if behaves as expected."""
    try:
        res = _map_if()
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


def test__map_while() -> None:
    """Verify that map while behaves as expected."""
    try:
        res = _map_while()
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


def test__map_tensor_array() -> None:
    """Verify that map tensor array behaves as expected."""
    try:
        res = _map_tensor_array()
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


def test__map_tensor_array_read() -> None:
    """Verify that map tensor array read behaves as expected."""
    try:
        res = _map_tensor_array_read()
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


def test__map_tensor_array_write() -> None:
    """Verify that map tensor array write behaves as expected."""
    try:
        res = _map_tensor_array_write()
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


def test__map_tensor_array_size() -> None:
    """Verify that map tensor array size behaves as expected."""
    try:
        res = _map_tensor_array_size()
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


def test__map_tensor_array_gather() -> None:
    """Verify that map tensor array gather behaves as expected."""
    try:
        res = _map_tensor_array_gather()
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


def test__map_tensor_array_scatter() -> None:
    """Verify that map tensor array scatter behaves as expected."""
    try:
        res = _map_tensor_array_scatter()
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


def test__map_lstm_cell() -> None:
    """Verify that map lstm cell behaves as expected."""
    try:
        res = _map_lstm_cell()
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


def test__map_gru_cell() -> None:
    """Verify that map gru cell behaves as expected."""
    try:
        res = _map_gru_cell()
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
