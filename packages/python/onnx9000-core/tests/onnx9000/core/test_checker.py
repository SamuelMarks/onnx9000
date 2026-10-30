"""Unit tests for ONNX model and tensor checker in `onnx9000.core.checker`."""

import pytest
from onnx9000.core.checker import (
    SchemaRegistry,
    ValidationContext,
    _check_op_specific,
    check_attribute,
    check_model,
    check_tensor,
)
from onnx9000.core.exceptions import UnsupportedOpError, UnsupportedOpsetError, ValidationError
from onnx9000.core.ir import Node, Tensor


def test_checker_functions() -> None:
    """Verify ValidationContext, SchemaRegistry, check_tensor, check_attribute, _check_op_specific, and check_model."""
    ctx = ValidationContext()
    assert ctx.strict is True

    reg = SchemaRegistry()
    assert "pads" in reg.get_schema("Conv", 13)
    with pytest.raises(UnsupportedOpsetError):
        reg.get_schema("Conv", 99)
    with pytest.raises(UnsupportedOpError):
        reg.get_schema("UnknownOp", 13)

    t = Tensor("t", (2, 2))
    t.data_type = "float"
    t.data_location = "DEFAULT"
    t.raw_data = b"\x00" * 16
    check_tensor(t, ctx)
    assert ctx.errors == []

    check_attribute("pads", [1, 1, 1, 1], "ints", ctx)
    assert ctx.errors == []

    node = Node("Add", ["a", "b"], ["c"])
    _check_op_specific(node, ctx)
    assert ctx.errors == []

    with pytest.raises(ValidationError):
        check_model(object())
