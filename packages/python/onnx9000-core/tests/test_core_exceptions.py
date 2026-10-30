"""Tests for ONNX9000 custom exceptions and unified error hierarchy."""

from onnx9000.core.exceptions import (
    AutogradError,
    BackendExecutionError,
    CLIError,
    CompilationError,
    ConversionError,
    GGUFError,
    Onnx9000Error,
    ONNXParseError,
    OptimizerError,
    SafetensorsError,
    ShapeInferenceError,
    ShapeMismatchError,
    TokenizationError,
    UnsupportedOpError,
    UnsupportedOpsetError,
    ValidationError,
)


def test_exception_hierarchy() -> None:
    """Verify that all domain exceptions inherit from Onnx9000Error and store messages."""
    err1 = UnsupportedOpError("UnknownOp")
    assert err1.op_type == "UnknownOp"
    assert err1.message == "Operator 'UnknownOp' is not supported yet."
    err2 = UnsupportedOpError("UnknownOp", "Custom message")
    assert err2.op_type == "UnknownOp"
    assert err2.message == "Custom message"

    subclasses = [
        Onnx9000Error,
        CompilationError,
        ShapeMismatchError,
        ONNXParseError,
        ShapeInferenceError,
        UnsupportedOpsetError,
        ValidationError,
        ConversionError,
        OptimizerError,
        AutogradError,
        BackendExecutionError,
        SafetensorsError,
        GGUFError,
        TokenizationError,
        CLIError,
    ]
    for cls in subclasses:
        inst = cls("test failure")
        assert isinstance(inst, Onnx9000Error)
        assert inst.message == "test failure"
        assert str(inst) == "test failure"
