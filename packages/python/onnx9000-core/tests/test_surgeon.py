"""Tests the surgeon module functionality."""

from onnx9000.core.ir import Graph, Node, Variable


def test_toposort_basic() -> None:
    """Tests the toposort basic functionality."""
    g = Graph("test")
    v1 = Variable("in1")
    v2 = Variable("in2")
    g.add_tensor(v1)
    g.add_tensor(v2)
    out1 = Variable("out1")
    g.add_tensor(out1)
    n1 = Node("Add", inputs=[v1, v2], outputs=[out1])
    g.add_node(n1)
    out2 = Variable("out2")
    g.add_tensor(out2)
    n2 = Node("Mul", inputs=[out1, v1], outputs=[out2])
    g.add_node(n2)
    g.nodes = [n2, n1]
    g.toposort()
    assert g.nodes == [n1, n2]
