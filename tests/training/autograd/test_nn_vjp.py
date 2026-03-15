"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.training.autograd.rules import get_vjp_rule


def test_gemm_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Gemm", ["a", "b", "c"], ["d"], {}, name="gemm_node")
    rule = get_vjp_rule("Gemm")
    nodes, names = rule.build_backward_nodes(node, ["grad_d"])
    assert len(nodes) == 3
    assert len(names) == 3


def test_convtranspose_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ConvTranspose", ["a", "w", "b"], ["c"], {}, name="convtrans_node")
    rule = get_vjp_rule("ConvTranspose")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 3
    assert len(names) == 3


def test_globalavgpool_vjp():
    """Provides semantic functionality and verification."""
    node = Node("GlobalAveragePool", ["a"], ["c"], {}, name="gap_node")
    rule = get_vjp_rule("GlobalAveragePool")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_globalmaxpool_vjp():
    """Provides semantic functionality and verification."""
    node = Node("GlobalMaxPool", ["a"], ["c"], {}, name="gmp_node")
    rule = get_vjp_rule("GlobalMaxPool")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_layernorm_vjp():
    """Provides semantic functionality and verification."""
    node = Node("LayerNormalization", ["a", "scale", "B"], ["c"], {}, name="ln_node")
    rule = get_vjp_rule("LayerNormalization")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert len(names) == 3


def test_instancenorm_vjp():
    """Provides semantic functionality and verification."""
    node = Node("InstanceNormalization", ["a", "scale", "B"], ["c"], {}, name="in_node")
    rule = get_vjp_rule("InstanceNormalization")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert len(names) == 3


def test_dropout_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Dropout", ["a"], ["c", "mask"], {}, name="dropout_node")
    rule = get_vjp_rule("Dropout")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
