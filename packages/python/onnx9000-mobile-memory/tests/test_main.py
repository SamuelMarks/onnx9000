"""Unit tests for onnx9000_mobile_memory re-exports."""

from onnx9000_mobile_memory import MobileMemory


def test_process() -> None:
    """Verify MobileMemory.process via onnx9000_mobile_memory."""
    assert MobileMemory().process("test") == "Mobile Memory processed test"
