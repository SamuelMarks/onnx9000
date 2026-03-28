"""Tests the infinite loop module functionality."""

from unittest.mock import patch

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier.api import simplify


def test_infinite_loop_break():
    """Tests the infinite loop break functionality."""
    g = Graph("mock")
    g.add_node(Node("Identity", ["a"], ["b"], {}, "id1"))
    g.add_node(Node("Identity", ["b"], ["c"], {}, "id2"))

    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    with patch.object(ConstantFoldingPass, "run") as mock_cf:

        def side_effect(graph):
            """Test the side effect functionality."""
            graph.add_node(Node("Identity", ["a"], ["b"], {}, "id1"))  # Fake nodes length changing

        mock_cf.side_effect = side_effect

        with patch("onnx9000.optimizer.simplifier.api.dead_code_elimination"):
            with patch("onnx9000.optimizer.simplifier.api.run_all_fusions"):
                simplify(g, max_iterations=3)
    assert len(g.nodes) == 5  # Initial 2 + 3 iterations
