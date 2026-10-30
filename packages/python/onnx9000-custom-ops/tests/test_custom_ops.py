"""Unit tests for onnx9000.custom_ops and onnx9000_custom_ops."""

import onnx9000_custom_ops
from onnx9000.custom_ops import CustomOpRegistry, registry


def test_custom_ops_registry() -> None:
    """Verify registering, retrieving, and listing custom operators."""

    def my_op() -> str:
        """Return confirmation string for custom op test."""
        return "ok"

    registry.register("MyOp", my_op)
    assert "MyOp" in registry.list_ops()
    op_fn = registry.get_op("MyOp")
    assert op_fn is not None
    assert op_fn() == "ok"
    assert registry.get_op("NonExistentOp") is None

    local_reg = onnx9000_custom_ops.CustomOpRegistry()
    assert isinstance(local_reg, CustomOpRegistry)
    assert local_reg.list_ops() == []
