"""Comprehensive unit tests for onnx9000.new_model_arch and onnx9000_new_model_arch."""

from onnx9000.new_model_arch import NewModelArch, resnet18, whisper_tiny


def test_new_model_arch_registry_and_builders() -> None:
    """Verify NewModelArch.process(), list_architectures(), and model builders."""
    n = NewModelArch()
    assert n.process("test") == "New Model Arch processed test"
    archs = n.list_architectures()
    assert "llama" in archs
    assert "whisper" in archs
    assert callable(resnet18)
    assert callable(whisper_tiny)
