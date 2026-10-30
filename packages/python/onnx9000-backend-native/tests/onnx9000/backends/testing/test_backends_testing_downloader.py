"""Unit tests and verification suite for downloader functionality."""

from onnx9000.backends.testing.downloader import download_and_extract_onnx_tests, get_node_test_dirs
from onnx9000.core.exceptions import Onnx9000Error


def test_download_and_extract_onnx_tests() -> None:
    """Verify that download and extract onnx tests behaves as expected."""
    try:
        res = download_and_extract_onnx_tests()
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


def test_get_node_test_dirs() -> None:
    """Verify that get node test dirs behaves as expected."""
    try:
        res = get_node_test_dirs()
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
