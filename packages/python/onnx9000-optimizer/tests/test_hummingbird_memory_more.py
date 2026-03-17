"""Tests the hummingbird memory more module functionality."""

import pytest
from onnx9000.optimizer.hummingbird.memory import (
    TreeAbstractions,
    estimate_memory_footprint,
    select_optimal_strategy,
)
from onnx9000.optimizer.hummingbird.strategies import Strategy, TargetHardware


def test_estimate_peak_vram():
    """Tests the estimate peak vram functionality."""
    t = TreeAbstractions()
    t.add_node(1, 0.5, 1, 2, 0.0)
    v1 = estimate_memory_footprint(t, Strategy.GEMM, 1)
    v2 = estimate_memory_footprint(t, Strategy.TREE_TRAVERSAL, 1)
    v3 = estimate_memory_footprint(t, Strategy.PERFECT_TREE_TRAVERSAL, 1)
    v0 = estimate_memory_footprint(t, "unknown", 1)

    assert v1 > 0
    assert v2 > 0
    assert v3 > 0
    assert v0 == 0

    t_empty = TreeAbstractions()
    v4 = estimate_memory_footprint(t_empty, Strategy.PERFECT_TREE_TRAVERSAL, 1)
    assert v4 == 0


def test_select_optimal_strategy():
    """Tests the select optimal strategy functionality."""
    t = TreeAbstractions()
    t.add_node(1, 0.5, 1, 2, 0.0)
    res = select_optimal_strategy(t, TargetHardware.WEBGPU)
    assert res == Strategy.GEMM

    t_deep = TreeAbstractions()
    for i in range(2000):  # deep tree log2(2000) > 10
        t_deep.add_node(i, 0.5, 1, 2, 0.0)
    res_deep = select_optimal_strategy(t_deep, TargetHardware.WEBGPU)
    assert res_deep == Strategy.GEMM

    res_gpu1 = select_optimal_strategy(t, TargetHardware.GPU, batch_size=1001)
    assert res_gpu1 == Strategy.GEMM
    res_gpu2 = select_optimal_strategy(t, TargetHardware.GPU, batch_size=10)
    assert res_gpu2 == Strategy.GEMM

    res_cpu1 = select_optimal_strategy(t, TargetHardware.CPU, batch_size=1)
    assert res_cpu1 == Strategy.TREE_TRAVERSAL
    res_cpu2 = select_optimal_strategy(t, TargetHardware.CPU, batch_size=10)
    assert res_cpu2 == Strategy.GEMM

    res_force = select_optimal_strategy(
        t, TargetHardware.CPU, force_strategy=Strategy.PERFECT_TREE_TRAVERSAL
    )
    assert res_force == Strategy.PERFECT_TREE_TRAVERSAL
