"""Unit tests for onnx9000_onnx_tool re-exports."""

from onnx9000_onnx_tool import ONNXTool


def test_process() -> None:
    """Verify ONNXTool.process via onnx9000_onnx_tool."""
    assert ONNXTool().process("test") == "ONNX Tool processed test"
