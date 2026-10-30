"""Unit tests for topological sort and cycle detection in `onnx9000.core.utils`."""

import pytest
from onnx9000.core.ir import Graph, Node
from onnx9000.core.utils import CyclicDependencyError, topological_sort


def test_topological_sort_and_cycles() -> None:
    """Verify topological_sort orders DAG nodes and raises CyclicDependencyError on cycles."""
    g = Graph("dag")
    n1 = Node("Relu", ["x"], ["m"], name="n1")
    n2 = Node("Sigmoid", ["m"], ["y"], name="n2")
    g.nodes = [n2, n1]

    ordered = topological_sort(g)
    assert ordered == [n1, n2]

    g_cycle = Graph("cycle")
    c1 = Node("Relu", ["b"], ["a"], name="c1")
    c2 = Node("Relu", ["a"], ["b"], name="c2")
    g_cycle.nodes = [c1, c2]
    with pytest.raises(CyclicDependencyError):
        topological_sort(g_cycle)
