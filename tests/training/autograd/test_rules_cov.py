"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node
from onnx9000.training.autograd.rules import LayerNormalizationVJP


def test_layernorm_with_bias():
    """Provides semantic logic and verification for LayerNorm with bias."""
    g = Graph("m")
    n = Node("LayerNormalization", ["x", "scale", "B"], ["y"], {})
    vjp = LayerNormalizationVJP()
    nodes, names = vjp.build_backward_nodes(n, ["grad_y"])
    assert len(nodes) == 1
    assert "LayerNormalizationGrad" == nodes[0].op_type
    assert len(names) == 3


def test_layernorm_no_bias():
    """Provides semantic logic and verification for LayerNorm no bias."""
    g = Graph("m")
    n = Node("LayerNormalization", ["x", "scale"], ["y"], {})
    vjp = LayerNormalizationVJP()
    nodes, names = vjp.build_backward_nodes(n, ["grad_y"])
    assert len(nodes) == 1
    assert "LayerNormalizationGrad" == nodes[0].op_type
    assert len(names) == 2


def test_get_vjp_rule_fallback():
    from onnx9000.training.autograd.rules import get_vjp_rule, ReluVJP

    rule = get_vjp_rule("UnknownOpTypeHere")
    assert isinstance(rule, ReluVJP)
