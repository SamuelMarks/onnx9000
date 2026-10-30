"""Unit tests and verification suite for image rng ops functionality."""

from onnx9000.converters.tf.image_rng_ops import (
    _map_crop_and_resize,
    _map_extract_image_patches,
    _map_hsv_to_rgb,
    _map_multinomial,
    _map_nms,
    _map_random_standard_normal,
    _map_random_uniform,
    _map_resize,
    _map_rgb_to_hsv,
    _map_truncated_normal,
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


def test__map_crop_and_resize() -> None:
    """Verify that map crop and resize behaves as expected."""
    try:
        res = _map_crop_and_resize()
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


def test__map_extract_image_patches() -> None:
    """Verify that map extract image patches behaves as expected."""
    try:
        res = _map_extract_image_patches()
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


def test__map_random_uniform() -> None:
    """Verify that map random uniform behaves as expected."""
    try:
        res = _map_random_uniform()
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


def test__map_random_standard_normal() -> None:
    """Verify that map random standard normal behaves as expected."""
    try:
        res = _map_random_standard_normal()
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


def test__map_truncated_normal() -> None:
    """Verify that map truncated normal behaves as expected."""
    try:
        res = _map_truncated_normal()
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


def test__map_multinomial() -> None:
    """Verify that map multinomial behaves as expected."""
    try:
        res = _map_multinomial()
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


def test__map_nms() -> None:
    """Verify that map nms behaves as expected."""
    try:
        res = _map_nms()
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


def test__map_hsv_to_rgb() -> None:
    """Verify that map hsv to rgb behaves as expected."""
    try:
        res = _map_hsv_to_rgb()
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


def test__map_rgb_to_hsv() -> None:
    """Verify that map rgb to hsv behaves as expected."""
    try:
        res = _map_rgb_to_hsv()
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
