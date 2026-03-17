from onnx9000.converters.paddle.passes import paddle_optimize_graph
from onnx9000.core.ir import Graph, Node, Tensor


def _create_graph() -> Graph:
    g = Graph(name="test")
    g.tensors["out"] = Tensor(name="out", dtype=1, shape=())
    g.outputs.append(g.tensors["out"])
    return g


def test_paddle_optimize_graph_identity_dropout_dce() -> None:
    g = _create_graph()
    n1 = Node(op_type="Identity", inputs=["in1"], outputs=["out1"], name="n1", attributes={})
    n2 = Node(op_type="Dropout", inputs=["out1"], outputs=["out2"], name="n2", attributes={})
    n3 = Node(op_type="Relu", inputs=["out2"], outputs=["out"], name="n3", attributes={})
    n4 = Node(op_type="Add", inputs=["out"], outputs=["dead_out"], name="n4", attributes={})
    n5 = Node(
        op_type="Custom_PaddleTest", inputs=["in1"], outputs=["out5"], name="n5", attributes={}
    )
    g.nodes = [n1, n2, n3, n4, n5]
    g = paddle_optimize_graph(g)
    assert len(g.nodes) == 2
    assert g.nodes[0].op_type == "Relu"
    assert g.nodes[0].inputs == ["in1"]
    assert g.nodes[1].op_type == "Custom_PaddleTest"
