"""Tests the profiler checks module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor
from onnx9000.core.profiler_checks import OptimizationAnalyzer


def test_profiler_checks_unused_initializers():
    """Tests the profiler checks unused initializers functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Constant("unused_weight", shape=(1,), dtype=DType.FLOAT32, values=b"\0" * 4))

    g.inputs.append("A")
    n = Node("Relu", inputs=["A"], outputs=["Y"])
    g.add_node(n)

    analyzer = OptimizationAnalyzer(g)
    opportunities = analyzer.analyze()

    found = any("unused initializers" in s for s in opportunities)
    assert found


def test_profiler_checks_matmul_add():
    """Tests the profiler checks matmul add functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(20, 30), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(30,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("Y", shape=(10, 30), dtype=DType.FLOAT32))

    g.inputs.extend(["A", "W", "B"])

    n1 = Node("MatMul", inputs=["A", "W"], outputs=["Y"])
    g.add_node(n1)
    n2 = Node("Add", inputs=["Y", "B"], outputs=["Z"])
    g.add_node(n2)

    analyzer = OptimizationAnalyzer(g)
    opportunities = analyzer.analyze()

    found = any("un-fused MatMul + Add" in s for s in opportunities)
    assert found


def test_profiler_checks_unsupported():
    """Tests the profiler checks unsupported functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10,), dtype=DType.FLOAT32))
    g.inputs.append("A")
    n = Node("NonZero", inputs=["A"], outputs=["Y"])
    g.add_node(n)

    analyzer = OptimizationAnalyzer(g)
    opportunities = analyzer.analyze()

    found = any("NonZero" in s and "unsupported" in s for s in opportunities)
    assert found
