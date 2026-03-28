"""Tests the base module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.base import Pass, PassContext


def test_pass_context() -> None:
    """Tests the pass context functionality."""
    ctx = PassContext("test")
    ctx.log_change("change")
    assert ctx.modifications == ["change"]


def test_pass_base() -> None:
    """Tests the base Pass functionality."""

    class MockPass(Pass):
        """Represents the MockPass class and its associated logic."""

        def run(self, graph):
            """Execute the run operation."""
            return super().run(graph)

    p = MockPass("test_pass")
    g = Graph("g")
    ctx = p.run(g)
    assert ctx.pass_name == "test_pass"
    assert repr(p) == "OptimizationPass(test_pass)"
