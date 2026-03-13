"""Module docstring."""

from onnx9000.autograd.optimizers import (
    add_sgd_optimizer,
    add_adam_optimizer,
    add_adamw_optimizer,
    add_gradient_accumulation,
    add_gradient_clipping,
)
from onnx9000.ir import Graph


def test_optimizers():
    """test_optimizers docstring."""
    graph = Graph("test")
    graph.initializers.append("param")
    graph.outputs.append("grad_param")

    add_sgd_optimizer(graph)
    assert len(graph.nodes) == 2

    add_adam_optimizer(graph)
    assert len(graph.nodes) == 3

    add_adamw_optimizer(graph)
    assert len(graph.nodes) == 4

    add_gradient_accumulation(graph)
    assert len(graph.nodes) == 5

    add_gradient_clipping(graph)
    assert len(graph.nodes) == 6
