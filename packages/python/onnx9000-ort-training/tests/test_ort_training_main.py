"""Unit tests for onnx9000_ort_training re-exports."""

from onnx9000_ort_training import ORTTraining


def test_process() -> None:
    """Verify ORTTraining.process via onnx9000_ort_training."""
    assert ORTTraining().process("test") == "ORT Training processed test"
