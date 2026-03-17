"""Tests the base cov final module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.base import GraphPass


def test_pass_base() -> None:
    """Tests the pass base functionality."""

    class DummyPass(GraphPass):
        """Represents the DummyPass class and its associated logic."""

        def run(self, graph: Graph) -> bool:
            """Tests the run functionality."""
            super().run(graph)
            return False

    p = DummyPass()
    p.run(Graph("test"))
