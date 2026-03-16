"""Module providing core logic and structural definitions."""

import onnx9000.toolkit.training.autograd.optimizers as opt
from onnx9000.core.ir import Graph


def test_sgd() -> None:
    """Tests the test_sgd functionality."""
    g = Graph("test")
    opt.add_sgd_optimizer(g, "lr", ["w1"], weight_decay=0.1, momentum=0.9)
    assert any((n.op_type == "Sub" and n.name == "w1_update_sub" for n in g.nodes))


def test_adam() -> None:
    """Tests the test_adam functionality."""
    g = Graph("test")
    opt.add_adam_optimizer(g, "lr", ["w1"])
    assert any((n.op_type == "AdamStep" for n in g.nodes))


def test_adamw() -> None:
    """Tests the test_adamw functionality."""
    g = Graph("test")
    opt.add_adamw_optimizer(g, "lr", ["w1"])
    assert any((n.op_type == "AdamWStep" for n in g.nodes))


def test_rmsprop() -> None:
    """Tests the test_rmsprop functionality."""
    g = Graph("test")
    opt.add_rmsprop_optimizer(g, "lr", ["w1"])
    assert any((n.op_type == "RMSpropStep" for n in g.nodes))


def test_adagrad() -> None:
    """Tests the test_adagrad functionality."""
    g = Graph("test")
    opt.add_adagrad_optimizer(g, "lr", ["w1"])
    assert any((n.op_type == "AdagradStep" for n in g.nodes))


def test_adadelta() -> None:
    """Tests the test_adadelta functionality."""
    g = Graph("test")
    opt.add_adadelta_optimizer(g, "lr", ["w1"])
    assert any((n.op_type == "AdadeltaStep" for n in g.nodes))
