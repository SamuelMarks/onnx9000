"""Comprehensive unit tests for onnx9000.onnx_checker and onnx9000_onnx_checker."""

import onnx9000_onnx_checker
from onnx9000.onnx_checker import SchemaRegistry, ValidationContext, check_model, check_tensor, run


def test_onnx_checker_run_and_registry() -> None:
    """Verify run(), SchemaRegistry, ValidationContext, and re-exports."""
    assert run() == "[onnx-checker] processed"
    assert onnx9000_onnx_checker.run() == "[onnx-checker] processed"

    ctx = ValidationContext(strict=True)
    assert ctx.errors == []
    reg = SchemaRegistry()
    assert "pads" in reg.get_schema("Conv", 13)
    assert callable(check_model)
    assert callable(check_tensor)
