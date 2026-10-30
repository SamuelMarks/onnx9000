"""Comprehensive unit tests for onnx9000.profiler and onnx9000_profiler."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.profiler import Profiler, ProfilerResult
from onnx9000_profiler import Profiler as LegacyProfiler


def test_profiler_default_and_graph() -> None:
    """Verify Profiler on both path-only and in-memory Graph targets."""
    p = LegacyProfiler("test.onnx")
    assert p.get_peak_memory() == 0.0
    assert p.run() is None
    assert p.get_peak_memory() == 42.5

    g = Graph("prof_graph")
    g.add_tensor(Tensor("X", (1024, 256), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1024, 256), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"]))

    gp = Profiler("in_mem.onnx", graph=g)
    res = gp.run()
    assert isinstance(res, ProfilerResult)
    assert gp.get_peak_memory() == 1.0
