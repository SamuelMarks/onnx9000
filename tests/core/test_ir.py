import pytest
from onnx9000.core.ir import DynamicDim, Tensor, Node, Graph
from onnx9000.core.dtypes import DType


def test_dynamic_dim():
    d1 = DynamicDim("batch")
    assert d1.value == "batch"
    assert repr(d1) == "DynamicDim(batch)"
    assert str(d1) == "batch"
    d2 = DynamicDim("batch")
    assert d1 == d2
    d3 = DynamicDim(-1)
    assert d1 != d3
    assert d1 != "batch"


def test_tensor():
    t = Tensor(
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
    assert repr(t).startswith("ir.Tensor(name=input_tensor")


def test_node():
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


def test_graph(caplog):
    g = Graph(name="test_graph")
    assert g.name == "test_graph"
    assert g.nodes == []
    assert g.tensors == {}
    t = Tensor(name="t1", shape=(1,), dtype=DType.FLOAT32)
    g.add_tensor(t)
    assert "t1" in g.tensors
    n = Node(op_type="Relu", inputs=["t1"], outputs=["t2"], attributes={})
    g.add_node(n)
    assert len(g.nodes) == 1
    g.inputs = ["t1"]
    g.outputs = ["t2"]
    import logging

    with caplog.at_level(logging.INFO):
        g.print_visualizer()
    assert "=== Graph: test_graph ===" in caplog.text
    assert "Inputs: ['t1']" in caplog.text
    assert "Outputs: ['t2']" in caplog.text
    assert "Relu: ['t1'] -> ['t2']" in caplog.text
