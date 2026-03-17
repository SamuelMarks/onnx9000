"""Tests the profiler checks more module functionality."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.profiler_checks import OptimizationAnalyzer


def test_optimization_checker():
    """Tests the optimization checker functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("x", [1], "float32"))
    g.add_tensor(Tensor("y", [1], "float32"))
    g.add_tensor(Tensor("z", [1], "float32"))
    g.add_node(Node("Cast", ["x"], ["y"], name="cast"))
    g.add_node(Node("Conv", ["x", "w"], ["conv_out"], name="conv"))
    g.add_tensor(Tensor("conv_out", [1], "float32"))
    g.add_node(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["bn_out"], name="bn"))
    g.add_node(Node("Identity", ["x"], ["x_id"], name="id"))

    checker = OptimizationAnalyzer(g)
    opts = checker.analyze()
    assert any("Conv + BatchNorm" in o for o in opts)
    assert any("Identity" in o for o in opts)


def test_optimization_checker_more():
    """Tests the optimization checker more functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("unused_init", [1], "float32", is_initializer=True))
    g.add_node(Node("MatMul", ["a", "b"], ["mm_out"], name="mm"))
    g.add_tensor(Tensor("mm_out", [1], "float32"))
    g.add_node(Node("Add", ["mm_out", "c"], ["add_out"], name="add"))
    g.add_node(Node("Loop", ["cond", "v_init"], ["v_out"], name="loop"))

    checker = OptimizationAnalyzer(g)
    opts = checker.analyze()
    assert any("unused initializers" in o for o in opts)
    assert any("MatMul + Add" in o for o in opts)
    assert any("Loop" in o for o in opts)
