"""Tests for fusions final."""

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.surgeon.fusions import fuse_horizontal_gemm


def test_fuse_horizontal_gemm_single() -> None:
    """Verify that fuse horizontal gemm single behaves as expected."""
    graph = Graph("test_graph")

    # Single Gemm
    n1 = Node("Gemm", ["X", "W1"], ["Y1"], {}, "gemm1")
    graph.nodes = [n1]

    # Should not crash and should not fuse
    fused = fuse_horizontal_gemm(graph)
    assert len(fused.nodes) == 1
    assert fused.nodes[0].op_type == "Gemm"
