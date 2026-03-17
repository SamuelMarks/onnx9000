"""Tests the compiler cov final module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import build_backward_graph


def test_autograd_compiler_missing() -> None:
    """Tests the autograd compiler missing functionality."""
    g = Graph("test")
    g.add_node(Node("Add", ["A", "B"], ["Y"], {}))
    g.add_tensor(Tensor("A", (1,), DType.FLOAT32, requires_grad=False))
    g.add_tensor(Tensor("B", (1,), DType.FLOAT32, requires_grad=True))
    g.add_tensor(Tensor("Y", (1,), DType.FLOAT32, requires_grad=True))
    g.outputs = ["Y"]
    bwd = build_backward_graph(g)
    assert "A_grad" not in bwd.tensors
    g.add_node(Node("Add", ["C", "C"], ["D"], {}))
    g.add_tensor(Tensor("C", (1,), DType.FLOAT32, requires_grad=True))
    g.add_tensor(Tensor("D", (1,), DType.FLOAT32, requires_grad=True))
    g.outputs = ["Y"]
    bwd2 = build_backward_graph(g)
    assert "C_grad" not in bwd2.tensors
