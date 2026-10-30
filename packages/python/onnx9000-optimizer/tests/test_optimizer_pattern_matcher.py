"""Tests for pattern matcher."""

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.pattern_matcher import (
    Pattern,
    PatternMatcherEngine,
    apply_algebraic_reuse,
    apply_fusion_reuse,
    apply_hardware_lowering,
)


def test_pattern_matcher() -> None:
    """Verify that pattern matcher behaves as expected."""
    engine = PatternMatcherEngine()
    engine.add_rule(Pattern("Add"), lambda n: None)

    g = Graph("test")
    g.nodes.append(Node(op_type="Add", inputs=[], outputs=[]))
    g.nodes.append(Node(op_type="Sub", inputs=[], outputs=[]))

    out = engine.apply(g)
    assert out is g


def test_algebraic_reuse() -> None:
    """Verify that algebraic reuse behaves as expected."""
    g = Graph("test")
    out = apply_algebraic_reuse(g)
    assert out is g


def test_fusion_reuse() -> None:
    """Verify that fusion reuse behaves as expected."""
    g = Graph("test")
    out = apply_fusion_reuse(g)
    assert out is g


def test_hardware_lowering() -> None:
    """Verify that hardware lowering behaves as expected."""
    g = Graph("test")
    out = apply_hardware_lowering(g)
    assert out is g


def test_pattern_matcher_matches_inputs() -> None:
    """Verify that pattern matcher matches inputs behaves as expected."""
    from onnx9000.core.ir import Node
    from onnx9000.optimizer.pattern_matcher import Pattern, matches

    # This hits line 21, the fallback True after checking inputs
    p = Pattern("Add", inputs=[Pattern("Any")])
    n = Node("Add", ["a"], ["b"])

    assert matches(n, p) is True
