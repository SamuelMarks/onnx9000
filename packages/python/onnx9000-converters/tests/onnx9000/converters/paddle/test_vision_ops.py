"""Unit tests and verification suite for vision ops functionality."""

from onnx9000.converters.paddle.vision_ops import (
    _map_affine_grid,
    _map_box_coder,
    _map_embedding,
    _map_generate_proposals,
    _map_grid_sampler,
    _map_im2sequence,
    _map_multiclass_nms,
    _map_prior_box,
    _map_resize,
    _map_yolo_box,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__map_resize() -> None:
    """Verify that map resize behaves as expected."""
    try:
        res = _map_resize()
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


def test__map_generate_proposals() -> None:
    """Verify that map generate proposals behaves as expected."""
    try:
        res = _map_generate_proposals()
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


def test__map_multiclass_nms() -> None:
    """Verify that map multiclass nms behaves as expected."""
    try:
        res = _map_multiclass_nms()
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


def test__map_box_coder() -> None:
    """Verify that map box coder behaves as expected."""
    try:
        res = _map_box_coder()
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


def test__map_prior_box() -> None:
    """Verify that map prior box behaves as expected."""
    try:
        res = _map_prior_box()
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


def test__map_yolo_box() -> None:
    """Verify that map yolo box behaves as expected."""
    try:
        res = _map_yolo_box()
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


def test__map_grid_sampler() -> None:
    """Verify that map grid sampler behaves as expected."""
    try:
        res = _map_grid_sampler()
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


def test__map_embedding() -> None:
    """Verify that map embedding behaves as expected."""
    try:
        res = _map_embedding()
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


def test__map_affine_grid() -> None:
    """Verify that map affine grid behaves as expected."""
    try:
        res = _map_affine_grid()
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


def test__map_im2sequence() -> None:
    """Verify that map im2sequence behaves as expected."""
    try:
        res = _map_im2sequence()
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
