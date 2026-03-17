import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.math_utils import (
    optimize_sigmoid,
    replace_mod,
    replace_where_with_arithmetic_mask,
    clamp_nan_to_zero,
    ensure_softmax_stability,
)


def test_optimize_sigmoid():
    g = Graph(name="test")
    optimize_sigmoid(g, "input", "out_sig", use_fast_math=True)
    op_types = [node.op_type for node in g.nodes]
    assert "Abs" in op_types
    assert "Add" in op_types
    assert "Div" in op_types


def test_replace_mod():
    g = Graph(name="test")
    replace_mod(g, "A", "B", "out_mod")
    op_types = [node.op_type for node in g.nodes]
    assert "Div" in op_types
    assert "Floor" in op_types
    assert "Mul" in op_types
    assert "Sub" in op_types


def test_replace_where_with_arithmetic_mask():
    g = Graph(name="test")
    replace_where_with_arithmetic_mask(g, "mask", "A", "B", "out")
    op_types = [node.op_type for node in g.nodes]
    assert op_types.count("Mul") == 2
    assert op_types.count("Sub") == 1
    assert op_types.count("Add") == 1


def test_clamp_nan_to_zero():
    g = Graph(name="test")
    clamp_nan_to_zero(g, "in", "out")
    op_types = [node.op_type for node in g.nodes]
    assert "IsNaN" in op_types
    assert "Where" in op_types


def test_ensure_softmax_stability():
    g = Graph(name="test")
    ensure_softmax_stability(g, "in", "out")
    op_types = [node.op_type for node in g.nodes]
    assert "ReduceMax" in op_types
    assert "Sub" in op_types
    assert "Exp" in op_types
    assert "ReduceSum" in op_types
    assert "Div" in op_types
