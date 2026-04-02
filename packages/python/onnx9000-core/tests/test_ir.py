"""Tests the ir module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import DynamicDim, Graph, Node, Variable
import sys

print(f"DEBUG sys.path: {sys.path}")
print(f"DEBUG Node class: {hex(id(Node))}")


def test_dynamic_dim() -> None:
    """Tests the dynamic dim functionality."""
    d1 = DynamicDim("batch")
    assert d1.value == "batch"
    assert repr(d1) == "DynamicDim(batch)"
    assert str(d1) == "batch"
    d2 = DynamicDim("batch")
    assert d1 == d2
    d3 = DynamicDim(-1)
    assert d1 != d3
    assert d1 != "batch"


def test_tensor() -> None:
    """Tests the tensor functionality."""
    t = Variable(
        name="input_tensor",
        shape=(DynamicDim("batch"), 3, 224, 224),
        dtype=DType.FLOAT32,
    )
    assert t.name == "input_tensor"
    assert t.shape[1] == 3
    assert t.dtype == DType.FLOAT32
    assert t.is_initializer is False
    assert t.requires_grad is True
    assert t.data is None
    assert repr(t).startswith("ir.Variable(name=input_tensor")


def test_node() -> None:
    """Tests the node functionality."""
    n = Node(
        op_type="Add",
        inputs=["a", "b"],
        outputs=["c"],
        attributes={"alpha": 1.0},
        name="add_node",
    )
    assert n.op_type == "Add"
    assert n.inputs == ["a", "b"]
    assert n.outputs == ["c"]
    assert n.attributes == {"alpha": 1.0}
    assert n.name == "add_node"
    assert repr(n) == "ir.Node(Add, ['a', 'b'] -> ['c'])"


def test_graph(caplog) -> None:
    """Tests the graph functionality."""
    g = Graph(name="test_graph")
    assert g.name == "test_graph"
    assert g.nodes == []
    assert g.tensors == {}
    t = Variable(name="t1", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(t)
    assert "t1" in g.tensors
    n = Node(op_type="Relu", inputs=["t1"], outputs=["t2"], attributes={})
    g.add_node(n)
    assert len(g.nodes) == 1

    class MockVInfo:
        """Represents the MockVInfo class and its associated logic."""

        def __init__(self, name) -> None:
            """Test the init   functionality."""
            self.name = name

    g.inputs = [MockVInfo("t1")]
    g.outputs = [MockVInfo("t2")]
    import logging

    with caplog.at_level(logging.INFO):
        g.print_visualizer()
    assert "=== Graph: test_graph ===" in caplog.text
    assert "Inputs: ['t1']" in caplog.text
    assert "Outputs: ['t2']" in caplog.text
    assert "Relu: ['t1'] -> ['t2']" in caplog.text

    # Test remove_node
    assert g.producer_map.get("t2") is n
    g.remove_node(n)
    assert len(g.nodes) == 0
    assert "t2" not in g.producer_map
    assert n not in g.consumer_map.get("t1", [])

    # Test equality
    g1 = Graph(name="g")
    g2 = Graph(name="g")
    assert g1 == g2
    g2.name = "g2"
    assert g1 != g2
    assert g1 != "not a graph"

    g1.name = "g"
    g2.name = "g"
    n1 = Node(op_type="Relu", inputs=["in"], outputs=["out"])
    g1.add_node(n1)
    assert g1 != g2
    n2 = Node(op_type="Relu", inputs=["in"], outputs=["out"])
    g2.add_node(n2)
    assert g1 == g2
