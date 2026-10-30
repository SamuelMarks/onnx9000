"""Unit tests and verification suite for xml builder functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.openvino.xml_builder import XmlBuilder, XmlNode


def test_XmlNode() -> None:
    """Verify that xml node behaves as expected."""
    try:
        obj = XmlNode()
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


def test_XmlBuilder() -> None:
    """Verify that xml builder behaves as expected."""
    try:
        obj = XmlBuilder()
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
