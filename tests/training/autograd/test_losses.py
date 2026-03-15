"""Tests the test losses functionality."""

from onnx9000.training.autograd.losses import add_mse_loss
from onnx9000.core.ir import Graph


def test_losses():
    """Tests the test losses functionality."""
    graph = Graph("test")
    add_mse_loss(graph, "pred", "target", "loss")
    assert len(graph.nodes) > 0
