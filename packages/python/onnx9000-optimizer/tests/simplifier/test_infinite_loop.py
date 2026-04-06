"""Tests the infinite loop module functionality."""

from unittest.mock import patch

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier import api
from onnx9000.optimizer.simplifier.api import simplify


def test_infinite_loop_break():
    """Tests the infinite loop break functionality."""
    g = Graph("mock")
    g.add_node(Node("Identity", ["a"], ["b"], {}, "id1"))
    g.add_node(Node("Identity", ["b"], ["c"], {}, "id2"))

    def side_effect(graph, *args, **kwargs):
        """Test the side effect functionality."""
        graph.add_node(Node("Identity", ["a"], ["b"], {}, f"id{len(graph.nodes)}"))

    from unittest.mock import MagicMock

    with patch.dict(
        simplify.__globals__,
        {
            "constant_folding": side_effect,
            "dead_code_elimination": MagicMock(),
            "run_all_fusions": MagicMock(),
        },
    ):
        simplify(g, max_iterations=3)
    assert len(g.nodes) == 5  # Initial 2 + 3 iterations
