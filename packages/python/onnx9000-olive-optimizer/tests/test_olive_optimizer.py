"""Comprehensive unit tests for onnx9000.olive_optimizer and onnx9000_olive_optimizer."""

from onnx9000.core.ir import Graph
from onnx9000.olive_optimizer import OliveModel, OliveOptimizer, Target


def test_olive_optimizer_pipeline() -> None:
    """Verify OliveOptimizer.process() and OliveOptimizer.optimize_model()."""
    o = OliveOptimizer()
    assert o.process("test") == "Olive Optimizer processed test"

    model = OliveModel(Graph("olive_test"))
    optimized = o.optimize_model(model, target=Target.CPU)
    assert isinstance(optimized, OliveModel)
