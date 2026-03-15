"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph
import onnx9000.training.autograd.optimizers as opt


def test_sgd():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_sgd_optimizer(g, "lr", ["w1"], weight_decay=0.1, momentum=0.9)
    assert any(n.op_type == "Sub" and n.name == "w1_update_sub" for n in g.nodes)


def test_adam():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_adam_optimizer(g, "lr", ["w1"])
    assert any(n.op_type == "AdamStep" for n in g.nodes)


def test_adamw():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_adamw_optimizer(g, "lr", ["w1"])
    assert any(n.op_type == "AdamWStep" for n in g.nodes)


def test_rmsprop():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_rmsprop_optimizer(g, "lr", ["w1"])
    assert any(n.op_type == "RMSpropStep" for n in g.nodes)


def test_adagrad():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_adagrad_optimizer(g, "lr", ["w1"])
    assert any(n.op_type == "AdagradStep" for n in g.nodes)


def test_adadelta():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    opt.add_adadelta_optimizer(g, "lr", ["w1"])
    assert any(n.op_type == "AdadeltaStep" for n in g.nodes)
