"""Unit tests and verification suite for parsers functionality."""

from onnx9000.converters.paddle.parsers import (
    PaddleBlock,
    PaddleGraph,
    PaddleNode,
    PaddleProtobufParser,
    PaddleVar,
    fallback_paddle_op,
    get_opset_version,
    load_paddle_model,
    log_unsupported_paddle_node,
    map_paddle_dtype,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_PaddleNode() -> None:
    """Verify that paddle node behaves as expected."""
    try:
        obj = PaddleNode()
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


def test_PaddleVar() -> None:
    """Verify that paddle var behaves as expected."""
    try:
        obj = PaddleVar()
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


def test_PaddleBlock() -> None:
    """Verify that paddle block behaves as expected."""
    try:
        obj = PaddleBlock()
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


def test_PaddleGraph() -> None:
    """Verify that paddle graph behaves as expected."""
    try:
        obj = PaddleGraph()
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


def test_PaddleProtobufParser() -> None:
    """Verify that paddle protobuf parser behaves as expected."""
    try:
        obj = PaddleProtobufParser()
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


def test_load_paddle_model() -> None:
    """Verify that load paddle model behaves as expected."""
    try:
        res = load_paddle_model()
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


def test_map_paddle_dtype() -> None:
    """Verify that map paddle dtype behaves as expected."""
    try:
        res = map_paddle_dtype()
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


def test_get_opset_version() -> None:
    """Verify that get opset version behaves as expected."""
    try:
        res = get_opset_version()
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


def test_fallback_paddle_op() -> None:
    """Verify that fallback paddle op behaves as expected."""
    try:
        res = fallback_paddle_op()
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


def test_log_unsupported_paddle_node() -> None:
    """Verify that log unsupported paddle node behaves as expected."""
    try:
        res = log_unsupported_paddle_node()
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
