"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph, Node
from onnx9000.toolkit.training.autograd.rules import LayerNormalizationVJP


def test_layernorm_with_bias() -> None:
    """Provides semantic logic and verification for LayerNorm with bias."""
    Graph("m")
    n = Node("LayerNormalization", ["x", "scale", "B"], ["y"], {})
    vjp = LayerNormalizationVJP()
    (nodes, names) = vjp.build_backward_nodes(n, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "LayerNormalizationGrad"
    assert len(names) == 3


def test_layernorm_no_bias() -> None:
    """Provides semantic logic and verification for LayerNorm no bias."""
    Graph("m")
    n = Node("LayerNormalization", ["x", "scale"], ["y"], {})
    vjp = LayerNormalizationVJP()
    (nodes, names) = vjp.build_backward_nodes(n, ["grad_y"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "LayerNormalizationGrad"
    assert len(names) == 2


def test_get_vjp_rule_fallback() -> None:
    """Tests the get vjp rule fallback functionality."""
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    rule = get_vjp_rule("UnknownOpTypeHere")
    assert rule is None
