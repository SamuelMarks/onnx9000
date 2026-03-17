"""Tests the optimizers module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.toolkit.training.autograd.optimizers import (
    add_adadelta_optimizer,
    add_adagrad_optimizer,
    add_adam_optimizer,
    add_adamw_optimizer,
    add_gradient_clipping,
    add_rmsprop_optimizer,
    add_sgd_optimizer,
)


def test_optimizers_coverage() -> None:
    """Tests the optimizers coverage functionality."""
    g = Graph("g")
    add_sgd_optimizer(g, "lr", ["w1"], weight_decay=0.1, momentum=0.9)
    add_adam_optimizer(g, "lr", ["w2"])
    add_adamw_optimizer(g, "lr", ["w3"], weight_decay=0.1)
    add_rmsprop_optimizer(g, "lr", ["w4"], weight_decay=0.1)
    add_adagrad_optimizer(g, "lr", ["w5"], weight_decay=0.1)
    add_adadelta_optimizer(g, "lr", ["w6"], weight_decay=0.1)
    assert len(g.nodes) > 10


def test_gradient_clipping() -> None:
    """Tests the gradient clipping functionality."""
    g = Graph("g")
    add_gradient_clipping(g, [], 1.0)
    names1 = ["grad_w1"]
    add_gradient_clipping(g, names1, 0.0)
    assert names1[0] == "grad_w1"
    add_gradient_clipping(g, names1, 1.0)
    assert names1[0] == "grad_w1_clipped"
    names2 = ["grad_w2", "grad_w3"]
    add_gradient_clipping(g, names2, 1.0)
    assert names2[0] == "grad_w2_clipped"


def test_optimizer_missing() -> None:
    """Tests the optimizer missing functionality."""
    from onnx9000.toolkit.training.autograd.optimizers import add_gradient_accumulation

    g = Graph("g")
    add_sgd_optimizer(g, "lr", ["w1"])
    add_gradient_accumulation(g, [], 1)
