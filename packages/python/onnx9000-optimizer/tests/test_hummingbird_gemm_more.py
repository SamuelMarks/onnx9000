"""Tests the hummingbird gemm more module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.engine import TreeAbstractions
from onnx9000.optimizer.hummingbird.gemm import (
    compile_decision_tree_classifier_gemm,
    compile_decision_tree_regressor_gemm,
    compile_isolation_forest_gemm,
    compile_partial_gemm,
    optimize_peak_vram_gemm,
)


def test_hummingbird_gemm_stubs():
    """Tests the hummingbird gemm stubs functionality."""
    g = Graph("g")
    t = TreeAbstractions()
    compile_partial_gemm(g, [t], 1)
    optimize_peak_vram_gemm([t])
    compile_decision_tree_regressor_gemm(g, t)
    compile_decision_tree_classifier_gemm(g, t)
    compile_isolation_forest_gemm(g, [t])
