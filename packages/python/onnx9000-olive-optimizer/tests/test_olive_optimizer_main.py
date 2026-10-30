"""Unit tests for onnx9000_olive_optimizer re-exports."""

from onnx9000_olive_optimizer import OliveOptimizer


def test_process() -> None:
    """Verify OliveOptimizer.process via onnx9000_olive_optimizer."""
    assert OliveOptimizer().process("test") == "Olive Optimizer processed test"
