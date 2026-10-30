"""Unit tests for Flax NNX codegen visitor in `onnx9000.core.codegen.flax`."""

from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, ValueInfo


def test_flax_visitor() -> None:
    """Verify ONNXToFlaxNNXVisitor generates valid Flax NNX module code."""
    g = Graph("flax_test")
    g.inputs.append(ValueInfo("x", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    code = ONNXToFlaxNNXVisitor(g).generate()
    assert "import jax" in code
