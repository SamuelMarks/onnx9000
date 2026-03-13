"""Module docstring."""

from onnx9000.autograd.losses import add_mse_loss, add_crossentropy_loss
from onnx9000.ir import Graph


def test_losses():
    """test_losses docstring."""
    graph = Graph("test")
    add_mse_loss(graph, "pred", "target")
    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "Sub"
    assert graph.nodes[1].op_type == "Mul"
    assert graph.nodes[2].op_type == "ReduceMean"

    add_crossentropy_loss(graph, "logits", "labels")
    assert len(graph.nodes) == 8
    assert graph.nodes[3].op_type == "Softmax"
    assert graph.nodes[4].op_type == "Log"
    assert graph.nodes[5].op_type == "Mul"
    assert graph.nodes[6].op_type == "Neg"
    assert graph.nodes[7].op_type == "ReduceMean"
