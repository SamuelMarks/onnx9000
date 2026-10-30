"""Comprehensive unit tests for onnx9000.ort_training and onnx9000_ort_training."""

from onnx9000.core.ir import Graph
from onnx9000.ort_training import AOTBuilder, ORTTraining, compile_training_graph


def test_ort_training_builder_and_process() -> None:
    """Verify ORTTraining.process(), create_builder(), and compile_training_graph export."""
    o = ORTTraining()
    assert o.process("test") == "ORT Training processed test"

    g = Graph("fwd")
    builder = o.create_builder(g)
    assert isinstance(builder, AOTBuilder)
    assert callable(compile_training_graph)
