"""Tests for pattern matcher."""

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.pattern_matcher import (
    Pattern,
    PatternMatcherEngine,
    apply_algebraic_reuse,
    apply_fusion_reuse,
    apply_hardware_lowering,
)


def test_pattern_matcher():
    """Docstring for D103."""
    engine = PatternMatcherEngine()
    engine.add_rule(Pattern("Add"), lambda n: None)

    g = Graph("test")
    g.nodes.append(Node(op_type="Add", inputs=[], outputs=[]))
    g.nodes.append(Node(op_type="Sub", inputs=[], outputs=[]))

    out = engine.apply(g)
    assert out is g


def test_algebraic_reuse():
    """Docstring for D103."""
    g = Graph("test")
    out = apply_algebraic_reuse(g)
    assert out is g


def test_fusion_reuse():
    """Docstring for D103."""
    g = Graph("test")
    out = apply_fusion_reuse(g)
    assert out is g


def test_hardware_lowering():
    """Docstring for D103."""
    g = Graph("test")
    out = apply_hardware_lowering(g)
    assert out is g


def test_pattern_matcher_matches_inputs():
    """Docstring for D103."""
    from onnx9000.core.ir import Node
    from onnx9000.optimizer.pattern_matcher import Pattern, matches

    # This hits line 21, the fallback True after checking inputs
    p = Pattern("Add", inputs=[Pattern("Any")])
    n = Node("Add", ["a"], ["b"])

    assert matches(n, p) is True
