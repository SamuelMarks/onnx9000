"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.training.autograd.rules import get_vjp_rule


def test_leakyrelu_vjp():
    """Provides semantic functionality and verification."""
    node = Node("LeakyRelu", ["a"], ["c"], {"alpha": 0.01}, name="leakyrelu_node")
    rule = get_vjp_rule("LeakyRelu")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "LeakyReluGrad"


def test_elu_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Elu", ["a"], ["c"], {"alpha": 1.0}, name="elu_node")
    rule = get_vjp_rule("Elu")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_selu_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Selu", ["a"], ["c"], {"alpha": 1.6, "gamma": 1.05}, name="selu_node")
    rule = get_vjp_rule("Selu")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_softplus_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Softplus", ["a"], ["c"], {}, name="softplus_node")
    rule = get_vjp_rule("Softplus")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_softsign_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Softsign", ["a"], ["c"], {}, name="softsign_node")
    rule = get_vjp_rule("Softsign")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_hardsigmoid_vjp():
    """Provides semantic functionality and verification."""
    node = Node(
        "HardSigmoid",
        ["a"],
        ["c"],
        {"alpha": 0.2, "beta": 0.5},
        name="hardsigmoid_node",
    )
    rule = get_vjp_rule("HardSigmoid")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_hardswish_vjp():
    """Provides semantic functionality and verification."""
    node = Node("HardSwish", ["a"], ["c"], {}, name="hardswish_node")
    rule = get_vjp_rule("HardSwish")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_gelu_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Gelu", ["a"], ["c"], {}, name="gelu_node")
    rule = get_vjp_rule("Gelu")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_softmax_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Softmax", ["a"], ["c"], {"axis": -1}, name="softmax_node")
    rule = get_vjp_rule("Softmax")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_logsoftmax_vjp():
    """Provides semantic functionality and verification."""
    node = Node("LogSoftmax", ["a"], ["c"], {"axis": -1}, name="logsoftmax_node")
    rule = get_vjp_rule("LogSoftmax")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_prelu_vjp():
    """Provides semantic functionality and verification."""
    node = Node("PRelu", ["a", "slope"], ["c"], {}, name="prelu_node")
    rule = get_vjp_rule("PRelu")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
