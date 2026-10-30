"""Unit tests for OptimizationAnalyzer in `onnx9000.core.profiler_checks`."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node
from onnx9000.core.profiler_checks import OptimizationAnalyzer


def test_optimization_analyzer() -> None:
    """Verify OptimizationAnalyzer detects unused initializers and identity ops."""
    g = Graph("opt_check")
    g.add_tensor(Constant("unused_w", values=b"\x00" * 4, shape=(1,), dtype=DType.FLOAT32))
    g.add_node(Node("Identity", ["x"], ["y"], name="id0"))

    analyzer = OptimizationAnalyzer(g)
    opps = analyzer.analyze()
    assert len(opps) >= 2
