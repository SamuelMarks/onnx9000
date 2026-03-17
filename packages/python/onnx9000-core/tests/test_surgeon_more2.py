"""Tests the surgeon more2 module functionality."""

from onnx9000.core.ir import Attribute, Constant, Graph, Node, Variable
from onnx9000.core.surgeon import cleanup, estimate_macs, fold_constants_math


def test_cleanup_producer_visited() -> None:
    """Tests the cleanup producer visited functionality."""
    g = Graph("clean")
    v_in = Variable("in")
    v_mid = Variable("mid")
    v_out = Variable("out")
    g.add_tensor(v_in)
    g.add_tensor(v_mid)
    g.add_tensor(v_out)
    g.promote_to_output(v_out)
    n1 = Node("N1", inputs=[v_in], outputs=[v_mid])
    n2 = Node("N2", inputs=[v_mid], outputs=[v_out])
    g.add_node(n1)
    g.add_node(n2)
    v_mid.inputs.append(n1)
    cleanup(g)
    assert n1 in g.nodes
    assert n2 in g.nodes


def test_estimate_macs_conv() -> None:
    """Tests the estimate macs conv functionality."""
    g = Graph("macs")
    v_in = Variable("in", shape=(1, 3, 224, 224))
    v_out = Variable("out", shape=(1, 64, 224, 224))
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    conv = Node("Conv", inputs=[v_in], outputs=[v_out])
    conv.attributes["kernel_shape"] = Attribute("kernel_shape", value=[3, 3])
    g.add_node(conv)
    macs = estimate_macs(g)
    assert macs == 64 * 224 * 224 * 3 * 3 * 3


def test_fold_constants_math_tensor_output() -> None:
    """Tests the fold constants math tensor output functionality."""
    g = Graph("fold")
    c1 = Constant("c1", values=b"1")
    c2 = Constant("c2", values=b"2")
    g.add_tensor(c1)
    g.add_tensor(c2)
    v_out = Variable("out")
    g.add_tensor(v_out)
    n_add = Node("Add", inputs=[c1, c2], outputs=[v_out])
    g.add_node(n_add)
    fold_constants_math(g)
    assert "out" in g.tensors
    assert isinstance(g.tensors["out"], Constant)


def test_estimate_macs_matmul_exception() -> None:
    """Tests the estimate macs matmul exception functionality."""
    g = Graph("mac")
    v1 = Variable("v1")
    n = Node("MatMul", inputs=[v1, v1])
    g.add_node(n)
    v1.shape = None
    assert estimate_macs(g) == 0


def test_deduplicate_constants_hash_collision_replace() -> None:
    """Tests the deduplicate constants hash collision replace functionality."""
    from onnx9000.core.surgeon import deduplicate_constants

    g = Graph("dedup")
    c1 = Constant("c1", values=b"dup")
    c2 = Constant("c2", values=b"dup")
    g.add_tensor(c1)
    g.add_tensor(c2)
    v_out = Variable("out")
    n = Node("N", inputs=[c1, c2], outputs=[v_out])
    g.add_node(n)
    c1.outputs.append(n)
    c2.outputs.append(n)
    deduplicate_constants(g)
    assert len(n.inputs) == 2
    assert n.inputs[0] == n.inputs[1]


def test_sink_transposes() -> None:
    """Tests the sink transposes functionality."""
    g = Graph("sink")
    v_in = Variable("in")
    v_mid = Variable("mid")
    v_out = Variable("out")
    n_trans = Node("Transpose", inputs=[v_in], outputs=[v_mid])
    g.add_node(n_trans)
    v_mid.inputs.append(n_trans)
    n_add = Node("Add", inputs=[v_mid], outputs=[v_out])
    g.add_node(n_add)
    from onnx9000.core.surgeon import sink_transposes

    sink_transposes(g)
