"""Unit tests for ONNX9000 graph profiler in `onnx9000.core.profiler`."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, DynamicDim, Graph, Node, Variable
from onnx9000.core.profiler import (
    ProfilerResult,
    _add_metric,
    dtype_size,
    get_attr,
    profile,
    profile_graph,
    resolve_volume,
)


def test_profiler_core_functions() -> None:
    """Verify ProfilerResult, dtype_size, resolve_volume, get_attr, _add_metric, profile, and profile_graph."""
    assert dtype_size(DType.FLOAT32) == 4
    assert resolve_volume((2, 3, 4)) == 24
    assert resolve_volume((DynamicDim("B"), 4), {"B": 2}) == 8

    node = Node("Conv", ["x", "w"], ["y"], attributes={"group": Attribute("group", "INT", 2)})
    assert get_attr(node, "group", 1) == 2
    assert get_attr(node, "missing", 5) == 5

    assert _add_metric(10, 20) == 30
    assert _add_metric("N", 10) == "(N + 10)"

    g = Graph("prof_test")
    g.add_tensor(Variable("x", (1, 3, 8, 8), DType.FLOAT32))
    g.add_tensor(Constant("w", values=b"\x00" * 108, shape=(1, 3, 3, 3), dtype=DType.FLOAT32))
    g.add_tensor(Variable("y", (1, 1, 6, 6), DType.FLOAT32))
    g.add_node(node)

    res1 = profile(g)
    assert isinstance(res1, ProfilerResult)
    assert res1.total_params == 27

    @profile_graph
    def _build(graph: Graph) -> Graph:
        """Execute build operation.

        Args:
            graph: Parameter specifying graph (Graph).

        Returns:
            Result of build (Graph).
        """
        return graph

    assert _build(g) is g
