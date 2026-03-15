import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.frontends.tf.passes import (
    constant_folding_pass,
    identity_removal_pass,
    dropout_removal_pass,
    remove_debug_nodes_pass,
    transpose_optimizer_pass,
    shape_folding_pass,
    pattern_matching_pass,
    dce_pass,
    tf_optimize_graph,
)


def _create_graph() -> Graph:
    g = Graph(name="test")
    g.tensors["out"] = Tensor(name="out", dtype=1, shape=())
    g.outputs.append(g.tensors["out"])
    return g


def test_constant_folding_pass():
    g = _create_graph()
    g = constant_folding_pass(g)
    assert g.name == "test"


def test_identity_removal_pass():
    g = _create_graph()
    n1 = Node(
        op_type="Relu", inputs=["in1"], outputs=["out1"], name="n1", attributes={}
    )
    n2 = Node(
        op_type="Identity", inputs=["out1"], outputs=["out2"], name="n2", attributes={}
    )
    n3 = Node(
        op_type="MatMul",
        inputs=["out2", "w"],
        outputs=["out"],
        name="n3",
        attributes={},
    )
    g.nodes = [n1, n2, n3]
    g = identity_removal_pass(g)
    assert len(g.nodes) == 2
    assert g.nodes[1].inputs == ["out1", "w"]


def test_identity_removal_pass_outputs():
    g = _create_graph()
    n1 = Node(
        op_type="Relu", inputs=["in1"], outputs=["out1"], name="n1", attributes={}
    )
    n2 = Node(
        op_type="Identity", inputs=["out1"], outputs=["out"], name="n2", attributes={}
    )
    g.nodes = [n1, n2]
    g = identity_removal_pass(g)
    assert len(g.nodes) == 1
    assert g.outputs[0].name == "out"


def test_dropout_removal_pass():
    g = _create_graph()
    n1 = Node(
        op_type="Dropout",
        inputs=["in1"],
        outputs=["out_drop", "mask"],
        name="n1",
        attributes={},
    )
    n2 = Node(
        op_type="Relu", inputs=["out_drop"], outputs=["out"], name="n2", attributes={}
    )
    g.nodes = [n1, n2]
    g = dropout_removal_pass(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].inputs == ["in1"]


def test_remove_debug_nodes_pass():
    g = _create_graph()
    n1 = Node(
        op_type="Custom_TFAssert",
        inputs=["in1"],
        outputs=["out_assert"],
        name="n1",
        attributes={},
    )
    n2 = Node(
        op_type="Relu", inputs=["out_assert"], outputs=["out"], name="n2", attributes={}
    )
    g.nodes = [n1, n2]
    g = remove_debug_nodes_pass(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].inputs == ["in1"]


def test_transpose_optimizer_pass():
    g = _create_graph()
    n1 = Node(
        op_type="Transpose",
        inputs=["in1"],
        outputs=["t1"],
        name="n1",
        attributes={"perm": [0, 2, 3, 1]},
    )
    n2 = Node(
        op_type="Transpose",
        inputs=["t1"],
        outputs=["t2"],
        name="n2",
        attributes={"perm": [0, 3, 1, 2]},
    )
    n3 = Node(op_type="Relu", inputs=["t2"], outputs=["out"], name="n3", attributes={})
    g.nodes = [n1, n2, n3]
    g = transpose_optimizer_pass(g)
    assert len(g.nodes) == 2
    assert g.nodes[1].inputs == ["in1"]


def test_shape_folding_pass():
    g = _create_graph()
    g = shape_folding_pass(g)
    assert g.name == "test"


def test_pattern_matching_pass():
    g = _create_graph()
    g = pattern_matching_pass(g)
    assert g.name == "test"


def test_dce_pass():
    g = _create_graph()
    n1 = Node(
        op_type="Relu", inputs=["in1"], outputs=["out1"], name="n1", attributes={}
    )
    n2 = Node(
        op_type="MatMul", inputs=["in2"], outputs=["out2"], name="n2", attributes={}
    )
    n3 = Node(
        op_type="Custom_TFPrint",
        inputs=["in3"],
        outputs=["out3"],
        name="n3",
        attributes={},
    )
    n4 = Node(op_type="Relu", inputs=["in4"], outputs=["out"], name="n4", attributes={})
    g.nodes = [n1, n2, n3, n4]
    g = dce_pass(g)
    assert len(g.nodes) == 2
    assert g.nodes[0].name == "n3"
    assert g.nodes[1].name == "n4"


def test_tf_optimize_graph():
    g = _create_graph()
    n1 = Node(
        op_type="Identity", inputs=["in1"], outputs=["out1"], name="n1", attributes={}
    )
    n2 = Node(
        op_type="Dropout",
        inputs=["out1"],
        outputs=["out2", "mask"],
        name="n2",
        attributes={},
    )
    n3 = Node(
        op_type="Relu", inputs=["out2"], outputs=["out"], name="n3", attributes={}
    )
    n4 = Node(
        op_type="Add", inputs=["out"], outputs=["dead_out"], name="n4", attributes={}
    )
    g.nodes = [n1, n2, n3, n4]
    g = tf_optimize_graph(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Relu"
    assert g.nodes[0].inputs == ["in1"]
