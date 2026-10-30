"""Unit tests for onnx9000_new_model_arch re-exports."""

from onnx9000_new_model_arch import NewModelArch


def test_process() -> None:
    """Verify NewModelArch.process via onnx9000_new_model_arch."""
    assert NewModelArch().process("test") == "New Model Arch processed test"
