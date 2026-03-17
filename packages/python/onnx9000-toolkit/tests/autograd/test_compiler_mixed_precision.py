from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import scale_backward_graph_for_mixed_precision


def test_scale_backward_graph_for_mixed_precision() -> None:
    g = Graph("test")
    g.initializers.append("w1")
    g.add_tensor(Tensor(name="w1", shape=(10,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor(name="grad_w1", shape=(10,), dtype="float32", requires_grad=True))
    g.add_node(Node("Identity", ["x"], ["grad_w1"]))
    g.add_node(Node("Mul", ["grad_w1", "lr"], ["w1_new"]))
    scale_backward_graph_for_mixed_precision(g, 1024.0)
    # Check that grad_w1 output was replaced by grad_w1_scaled
    assert any("grad_w1_scaled" in n.outputs for n in g.nodes)
    assert any(n.op_type == "Div" and n.outputs == ["grad_w1"] for n in g.nodes)
