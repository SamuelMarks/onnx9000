"""Unit tests and verification suite for exceptions functionality."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))

import onnx9000.core.exceptions as exc


def test_Onnx9000Error() -> None:
    """Verify that onnx9000 error behaves as expected."""
    obj = exc.Onnx9000Error()
    assert obj.args[0] == "An error occurred"


def test_CompilationError() -> None:
    """Verify that compilation error behaves as expected."""
    obj = exc.CompilationError()
    assert obj.args[0] == "An error occurred"


def test_UnsupportedOpError() -> None:
    """Verify that unsupported op error behaves as expected."""
    obj = exc.UnsupportedOpError("TestOp")
    assert obj.op_type == "TestOp"
    assert obj.message == "Operator 'TestOp' is not supported yet."


def test_ShapeMismatchError() -> None:
    """Verify that shape mismatch error behaves as expected."""
    obj = exc.ShapeMismatchError()
    assert obj is not None


def test_ONNXParseError() -> None:
    """Verify that onnxparse error behaves as expected."""
    obj = exc.ONNXParseError()
    assert obj is not None


def test_ShapeInferenceError() -> None:
    """Verify that shape inference error behaves as expected."""
    obj = exc.ShapeInferenceError()
    assert obj is not None


def test_UnsupportedOpsetError() -> None:
    """Verify that unsupported opset error behaves as expected."""
    obj = exc.UnsupportedOpsetError()
    assert obj is not None


def test_ValidationError() -> None:
    """Verify that validation error behaves as expected."""
    obj = exc.ValidationError()
    assert obj is not None
