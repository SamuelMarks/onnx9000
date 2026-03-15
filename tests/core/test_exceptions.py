"""Tests for ONNX9000 custom exceptions and error handling."""

from onnx9000.core.exceptions import UnsupportedOpError


def test_unsupported_op_error():
    """Verify that UnsupportedOpError initializes correctly with and without a custom message."""
    err1 = UnsupportedOpError("UnknownOp")
    assert err1.op_type == "UnknownOp"
    assert err1.message == "Operator 'UnknownOp' is not supported yet."
    err2 = UnsupportedOpError("UnknownOp", "Custom message")
    assert err2.op_type == "UnknownOp"
    assert err2.message == "Custom message"
