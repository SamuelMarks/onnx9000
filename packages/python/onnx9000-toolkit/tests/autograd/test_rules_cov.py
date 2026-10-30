"""Unit tests and verification suite for rules cov functionality."""

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


def test_conv_pool_and_linalg_vjp_branches() -> None:
    """Verify ConvTransposeVJP, ResizeVJP, GemmVJP, and TriluVJP optional input branches."""
    from onnx9000.toolkit.training.autograd.rules import (
        ConvTransposeVJP,
        GemmVJP,
        ResizeVJP,
        TriluVJP,
    )

    ct_no_bias = Node("ConvTranspose", ["x", "w"], ["y"], {}, name="ct")
    nodes, names = ConvTransposeVJP().build_backward_nodes(ct_no_bias, ["gy"])
    assert len(nodes) == 2
    assert len(names) == 2

    resize_cubic = Node("Resize", ["x", "roi", "scales"], ["y"], {"mode": "cubic"}, name="rsz")
    nodes_r, names_r = ResizeVJP().build_backward_nodes(resize_cubic, ["gy"])
    assert len(nodes_r) == 1
    assert nodes_r[0].op_type == "Identity"
    assert len(names_r) == 3

    gemm_no_bias = Node("Gemm", ["a", "b"], ["y"], {}, name="gemm")
    nodes_g, names_g = GemmVJP().build_backward_nodes(gemm_no_bias, ["gy"])
    assert len(nodes_g) == 2
    assert len(names_g) == 2

    trilu_with_k = Node("Trilu", ["x", "k"], ["y"], {}, name="trilu")
    nodes_t, names_t = TriluVJP().build_backward_nodes(trilu_with_k, ["gy"])
    assert len(nodes_t) == 1
    assert nodes_t[0].inputs == ["gy", "k"]
    assert len(names_t) == 2

    trilu_no_k = Node("Trilu", ["x"], ["y"], {}, name="trilu1")
    nodes_t1, names_t1 = TriluVJP().build_backward_nodes(trilu_no_k, ["gy"])
    assert len(nodes_t1) == 1
    assert nodes_t1[0].inputs == ["gy"]
    assert len(names_t1) == 1
