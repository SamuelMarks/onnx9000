import pytest
from onnx9000.optimizer.hummingbird import (
    TranspilationEngine,
    Strategy,
    TargetHardware,
    TreeAbstractions,
    estimate_memory_footprint,
    select_optimal_strategy,
)
from onnx9000.core.ir import Graph


def test_strategy_enum():
    assert Strategy.GEMM
    assert Strategy.TREE_TRAVERSAL
    assert Strategy.PERFECT_TREE_TRAVERSAL
    assert TargetHardware.CPU
    assert TargetHardware.GPU
    assert TargetHardware.WEBGPU


def test_tree_abstractions():
    tree = TreeAbstractions()
    tree.add_node(feature=0, threshold=1.5, left=1, right=2, value=0.0)
    assert len(tree.features) == 1
    assert tree.features[0] == 0
    assert tree.thresholds[0] == 1.5
    assert tree.left_children[0] == 1
    assert tree.right_children[0] == 2
    assert tree.values[0] == 0.0


def test_memory_estimator():
    tree = TreeAbstractions()
    for i in range(10):
        tree.add_node(i, 0.5, i + 1, i + 2, 1.0)

    mem_gemm = estimate_memory_footprint(tree, Strategy.GEMM, batch_size=1)
    # A=10*10*4, B=10*4, C=10*4 -> 400 + 40 + 40 = 480
    assert mem_gemm == 480

    mem_tree = estimate_memory_footprint(tree, Strategy.TREE_TRAVERSAL, batch_size=1)
    # 10 * 6 * 4 = 240
    assert mem_tree == 240


def test_strategy_selector():
    tree = TreeAbstractions()
    for i in range(10):
        tree.add_node(i, 0.5, i + 1, i + 2, 1.0)

    # Force strategy
    strat = select_optimal_strategy(
        tree, TargetHardware.CPU, force_strategy=Strategy.PERFECT_TREE_TRAVERSAL
    )
    assert strat == Strategy.PERFECT_TREE_TRAVERSAL

    # Auto CPU batch 1 -> TreeTraversal
    strat = select_optimal_strategy(tree, TargetHardware.CPU, batch_size=1)
    assert strat == Strategy.TREE_TRAVERSAL

    # Auto WebGPU -> GEMM
    strat = select_optimal_strategy(tree, TargetHardware.WEBGPU, batch_size=1)
    assert strat == Strategy.GEMM


def test_transpilation_engine():
    engine = TranspilationEngine(TargetHardware.CPU)
    # Empty abstractions
    g = engine.transpile("fake_model")
    assert isinstance(g, Graph)

    # With abstractions
    tree = TreeAbstractions()
    tree.add_node(0, 0.5, -1, -1, 1.0)
    engine.abstractions.append(tree)
    g = engine.transpile("fake_model", force_strategy=Strategy.GEMM)
    assert isinstance(g, Graph)


def test_engine_verbose():
    from onnx9000.optimizer.hummingbird.engine import TranspilationEngine
    from onnx9000.optimizer.hummingbird.engine import TreeAbstractions

    engine = TranspilationEngine("webgpu", verbose=True)
    engine.register_backend("test", None)

    ab = TreeAbstractions()
    ab.add_node(0, 0.5, 1, 2, 0.0)
    ab.add_node(0, 0.0, -1, -1, 1.0)
    ab.add_node(0, 0.0, -1, -1, 2.0)
    engine.abstractions = [ab]

    engine.transpile(None)
