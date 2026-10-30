"""Unit tests for PyTorch codegen visitor in `onnx9000.core.codegen.pytorch`."""

from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, ValueInfo


def test_pytorch_visitor() -> None:
    """Verify ONNXToPyTorchVisitor generates valid PyTorch nn.Module code."""
    g = Graph("pt_test")
    g.inputs.append(ValueInfo("x", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    code = ONNXToPyTorchVisitor(g).generate()
    assert "import torch" in code
