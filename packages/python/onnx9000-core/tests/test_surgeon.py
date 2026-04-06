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
    #
    assert len(g.nodes) == 2


def test_surgeon_missing_lines():
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.surgeon import map_alibi, map_gqa_mqa, visualize_browser_canvas

    g = Graph("test")
    g.nodes.append(Node("ALiBi", inputs=["a", "b"], outputs=["c"], name="alibi"))
    g2 = map_alibi(g)
    assert len(g2.nodes) == 2
    assert g2.nodes[0].op_type == "Add"
    assert g2.nodes[1].op_type == "Mask"

    g3 = Graph("test")
    g3.nodes.append(
        Node(
            "GQA", inputs=["a", "b", "c"], outputs=["d"], attributes={"num_heads": 8, "kv_heads": 8}
        )
    )
    g4 = map_gqa_mqa(g3)
    assert g4.nodes[0].op_type == "MultiHeadAttention"

    g5 = Graph("test")
    g5.nodes.append(
        Node(
            "GQA", inputs=["a", "b", "c"], outputs=["d"], attributes={"num_heads": 8, "kv_heads": 1}
        )
    )
    g6 = map_gqa_mqa(g5)
    assert g6.nodes[0].op_type == "MultiHeadAttention"

    visualize_browser_canvas(g, "abc")
