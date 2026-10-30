"""Unit tests for Keras codegen visitor in `onnx9000.core.codegen.keras`."""

from onnx9000.core.codegen.keras import ONNXToKerasVisitor
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, ValueInfo


def test_keras_visitor() -> None:
    """Verify ONNXToKerasVisitor generates valid Keras module code."""
    g = Graph("keras_test")
    g.inputs.append(ValueInfo("x", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    code = ONNXToKerasVisitor(g).generate()
    assert "import keras" in code
