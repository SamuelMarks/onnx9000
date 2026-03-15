"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.training.autograd.rules import get_vjp_rule


def test_sub_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Sub", ["a", "b"], ["c"], {}, name="sub_node")
    rule = get_vjp_rule("Sub")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "Identity"
    assert nodes[1].op_type == "Neg"
    assert names == ["grad_a_wrt_sub_node", "grad_b_wrt_sub_node"]


def test_div_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Div", ["a", "b"], ["c"], {}, name="div_node")
    rule = get_vjp_rule("Div")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 5
    assert names == ["grad_a_wrt_div_node", "grad_b_wrt_div_node"]


def test_pow_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Pow", ["a", "b"], ["c"], {}, name="pow_node")
    rule = get_vjp_rule("Pow")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_mod_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Mod", ["a", "b"], ["c"], {}, name="mod_node")
    rule = get_vjp_rule("Mod")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_abs_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Abs", ["a"], ["c"], {}, name="abs_node")
    rule = get_vjp_rule("Abs")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_neg_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Neg", ["a"], ["c"], {}, name="neg_node")
    rule = get_vjp_rule("Neg")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sign_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Sign", ["a"], ["c"], {}, name="sign_node")
    rule = get_vjp_rule("Sign")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_exp_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Exp", ["a"], ["c"], {}, name="exp_node")
    rule = get_vjp_rule("Exp")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_log_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Log", ["a"], ["c"], {}, name="log_node")
    rule = get_vjp_rule("Log")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sqrt_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Sqrt", ["a"], ["c"], {}, name="sqrt_node")
    rule = get_vjp_rule("Sqrt")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sin_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Sin", ["a"], ["c"], {}, name="sin_node")
    rule = get_vjp_rule("Sin")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_cos_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Cos", ["a"], ["c"], {}, name="cos_node")
    rule = get_vjp_rule("Cos")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 3


def test_tan_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Tan", ["a"], ["c"], {}, name="tan_node")
    rule = get_vjp_rule("Tan")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_asin_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Asin", ["a"], ["c"], {}, name="asin_node")
    rule = get_vjp_rule("Asin")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_acos_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Acos", ["a"], ["c"], {}, name="acos_node")
    rule = get_vjp_rule("Acos")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_atan_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Atan", ["a"], ["c"], {}, name="atan_node")
    rule = get_vjp_rule("Atan")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sinh_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Sinh", ["a"], ["c"], {}, name="sinh_node")
    rule = get_vjp_rule("Sinh")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_cosh_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Cosh", ["a"], ["c"], {}, name="cosh_node")
    rule = get_vjp_rule("Cosh")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_asinh_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Asinh", ["a"], ["c"], {}, name="asinh_node")
    rule = get_vjp_rule("Asinh")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_acosh_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Acosh", ["a"], ["c"], {}, name="acosh_node")
    rule = get_vjp_rule("Acosh")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_atanh_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Atanh", ["a"], ["c"], {}, name="atanh_node")
    rule = get_vjp_rule("Atanh")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_erf_vjp():
    """Provides semantic functionality and verification."""
    node = Node("Erf", ["a"], ["c"], {}, name="erf_node")
    rule = get_vjp_rule("Erf")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_isnan_vjp():
    """Provides semantic functionality and verification."""
    node = Node("IsNaN", ["a"], ["c"], {}, name="isnan_node")
    rule = get_vjp_rule("IsNaN")
    nodes, names = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
