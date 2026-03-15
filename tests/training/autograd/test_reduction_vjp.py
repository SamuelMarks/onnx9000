"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.training.autograd.rules import get_vjp_rule


def test_reducesum_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceSum", ["a"], ["c"], {"keepdims": 1}, name="reducesum_node")
    rule = get_vjp_rule("ReduceSum")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "Expand"


def test_reducemean_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceMean", ["a"], ["c"], {"keepdims": 1}, name="reducemean_node")
    rule = get_vjp_rule("ReduceMean")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "Expand"
    assert nodes[1].op_type == "Div"


def test_reducemax_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceMax", ["a"], ["c"], {}, name="reducemax_node")
    rule = get_vjp_rule("ReduceMax")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducemin_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceMin", ["a"], ["c"], {}, name="reducemin_node")
    rule = get_vjp_rule("ReduceMin")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reduceprod_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceProd", ["a"], ["c"], {}, name="reduceprod_node")
    rule = get_vjp_rule("ReduceProd")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducel1_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceL1", ["a"], ["c"], {}, name="reducel1_node")
    rule = get_vjp_rule("ReduceL1")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducel2_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceL2", ["a"], ["c"], {}, name="reducel2_node")
    rule = get_vjp_rule("ReduceL2")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducelogsum_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceLogSum", ["a"], ["c"], {}, name="reducelogsum_node")
    rule = get_vjp_rule("ReduceLogSum")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducelogsumexp_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceLogSumExp", ["a"], ["c"], {}, name="reducelogsumexp_node")
    rule = get_vjp_rule("ReduceLogSumExp")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reducesumsquare_vjp():
    """Provides semantic functionality and verification."""
    node = Node("ReduceSumSquare", ["a"], ["c"], {}, name="reducesumsquare_node")
    rule = get_vjp_rule("ReduceSumSquare")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
