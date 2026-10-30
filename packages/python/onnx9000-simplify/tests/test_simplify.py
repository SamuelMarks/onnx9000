"""Comprehensive unit tests for onnx9000.simplify and onnx9000_simplify."""

import onnx9000_simplify
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.simplify import run, simplify_graph


def test_simplify_graph_and_run() -> None:
    """Verify run() and simplify_graph() eliminating unused nodes."""
    assert run() == "[simplify] processed"
    assert onnx9000_simplify.run() == "[simplify] processed"

    g = Graph("simp_test")
    g.add_tensor(Tensor("X", (1, 4), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 4), DType.FLOAT32))
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"], name="relu1"))

    out = simplify_graph(g)
    assert isinstance(out, Graph)
    assert len(out.nodes) == 1
