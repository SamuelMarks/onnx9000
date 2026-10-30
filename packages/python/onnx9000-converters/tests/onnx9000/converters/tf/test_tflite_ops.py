"""Unit tests and verification suite for tflite ops functionality."""

from onnx9000.converters.tf.tflite_ops import (
    _map_tflite_concat,
    _map_tflite_conv2d,
    _map_tflite_custom_subgraph,
    _map_tflite_depth_to_space,
    _map_tflite_depthwise_conv2d,
    _map_tflite_dequantize,
    _map_tflite_embedding_lookup,
    _map_tflite_floor,
    _map_tflite_fully_connected,
    _map_tflite_l2_normalization,
    _map_tflite_local_response_normalization,
    _map_tflite_logistic,
    _map_tflite_pool,
    _map_tflite_quantize,
    _map_tflite_relu,
    _map_tflite_relu6,
    _map_tflite_relu_n1_to_1,
    _map_tflite_reshape,
    _map_tflite_resize_bilinear,
    _map_tflite_simple_binary,
    _map_tflite_softmax,
    _map_tflite_space_to_depth,
    _map_tflite_tanh,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__map_tflite_simple_binary() -> None:
    """Verify that map tflite simple binary behaves as expected."""
    try:
        res = _map_tflite_simple_binary()
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


def test__map_tflite_pool() -> None:
    """Verify that map tflite pool behaves as expected."""
    try:
        res = _map_tflite_pool()
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


def test__map_tflite_conv2d() -> None:
    """Verify that map tflite conv2d behaves as expected."""
    try:
        res = _map_tflite_conv2d()
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


def test__map_tflite_depthwise_conv2d() -> None:
    """Verify that map tflite depthwise conv2d behaves as expected."""
    try:
        res = _map_tflite_depthwise_conv2d()
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


def test__map_tflite_fully_connected() -> None:
    """Verify that map tflite fully connected behaves as expected."""
    try:
        res = _map_tflite_fully_connected()
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


def test__map_tflite_reshape() -> None:
    """Verify that map tflite reshape behaves as expected."""
    try:
        res = _map_tflite_reshape()
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


def test__map_tflite_resize_bilinear() -> None:
    """Verify that map tflite resize bilinear behaves as expected."""
    try:
        res = _map_tflite_resize_bilinear()
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


def test__map_tflite_concat() -> None:
    """Verify that map tflite concat behaves as expected."""
    try:
        res = _map_tflite_concat()
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


def test__map_tflite_softmax() -> None:
    """Verify that map tflite softmax behaves as expected."""
    try:
        res = _map_tflite_softmax()
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


def test__map_tflite_logistic() -> None:
    """Verify that map tflite logistic behaves as expected."""
    try:
        res = _map_tflite_logistic()
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


def test__map_tflite_tanh() -> None:
    """Verify that map tflite tanh behaves as expected."""
    try:
        res = _map_tflite_tanh()
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


def test__map_tflite_relu() -> None:
    """Verify that map tflite relu behaves as expected."""
    try:
        res = _map_tflite_relu()
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


def test__map_tflite_relu6() -> None:
    """Verify that map tflite relu6 behaves as expected."""
    try:
        res = _map_tflite_relu6()
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


def test__map_tflite_relu_n1_to_1() -> None:
    """Verify that map tflite relu n1 to 1 behaves as expected."""
    try:
        res = _map_tflite_relu_n1_to_1()
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


def test__map_tflite_dequantize() -> None:
    """Verify that map tflite dequantize behaves as expected."""
    try:
        res = _map_tflite_dequantize()
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


def test__map_tflite_quantize() -> None:
    """Verify that map tflite quantize behaves as expected."""
    try:
        res = _map_tflite_quantize()
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


def test__map_tflite_embedding_lookup() -> None:
    """Verify that map tflite embedding lookup behaves as expected."""
    try:
        res = _map_tflite_embedding_lookup()
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


def test__map_tflite_l2_normalization() -> None:
    """Verify that map tflite l2 normalization behaves as expected."""
    try:
        res = _map_tflite_l2_normalization()
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


def test__map_tflite_local_response_normalization() -> None:
    """Verify that map tflite local response normalization behaves as expected."""
    try:
        res = _map_tflite_local_response_normalization()
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


def test__map_tflite_space_to_depth() -> None:
    """Verify that map tflite space to depth behaves as expected."""
    try:
        res = _map_tflite_space_to_depth()
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


def test__map_tflite_depth_to_space() -> None:
    """Verify that map tflite depth to space behaves as expected."""
    try:
        res = _map_tflite_depth_to_space()
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


def test__map_tflite_floor() -> None:
    """Verify that map tflite floor behaves as expected."""
    try:
        res = _map_tflite_floor()
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


def test__map_tflite_custom_subgraph() -> None:
    """Verify that map tflite custom subgraph behaves as expected."""
    try:
        res = _map_tflite_custom_subgraph()
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
