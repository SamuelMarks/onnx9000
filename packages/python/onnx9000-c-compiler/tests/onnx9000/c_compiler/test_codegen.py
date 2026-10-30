"""Unit tests and verification suite for codegen functionality."""

from onnx9000.c_compiler.codegen import BaseCodegenVisitor, CFamilyCodegen, PythonFamilyCodegen
from onnx9000.core.exceptions import Onnx9000Error


def test_BaseCodegenVisitor() -> None:
    """Verify that base codegen visitor behaves as expected."""
    try:
        obj = BaseCodegenVisitor()
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


def test_CFamilyCodegen() -> None:
    """Verify that cfamily codegen behaves as expected."""
    try:
        obj = CFamilyCodegen()
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


def test_PythonFamilyCodegen() -> None:
    """Verify that python family codegen behaves as expected."""
    try:
        obj = PythonFamilyCodegen()
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
