"""Tests the test optimizers functionality."""

from onnx9000.training.autograd.optimizers import (
    add_sgd_optimizer,
    add_gradient_accumulation,
    add_gradient_clipping,
)
from onnx9000.core.ir import Graph


def test_optimizers():
    """Tests the test optimizers functionality."""
    graph = Graph("test")
    graph.initializers.append("param")
    graph.outputs.append("grad_param")
    add_sgd_optimizer(graph, "lr", ["param"])
    add_gradient_accumulation(graph, ["grad_param"], 2)
    add_gradient_clipping(graph, ["grad_param"], 1.0)
    assert len(graph.nodes) > 0
