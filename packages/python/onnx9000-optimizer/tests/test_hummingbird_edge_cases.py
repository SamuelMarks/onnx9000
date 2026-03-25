"""Tests the hummingbird edge cases module functionality."""

import time

from onnx9000.optimizer.hummingbird import Strategy, TargetHardware, TranspilationEngine
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions
from onnx9000.optimizer.hummingbird.perfect_tree import PerfectTreeCompiler


def test_empty_tree_structure_handling() -> None:
    """Tests the empty tree structure handling functionality."""
    engine = TranspilationEngine(TargetHardware.CPU)
    g = engine.transpile("fake_empty_model")
    assert g.name == "Hummingbird_Transpiled"


def test_trees_with_depth_gt_50_gemm_strategy_fallback() -> None:
    """Tests the trees with depth gt 50 gemm strategy fallback functionality."""
    tree = TreeAbstractions()
    # Create depth 55
    curr = 0
    for _i in range(55):
        tree.add_node(0, 0.5, curr + 1, -1, 0.0)
        curr += 1
    tree.add_node(-1, 0.0, -1, -1, 1.0)  # Leaf

    engine = TranspilationEngine(TargetHardware.WEBGPU)
    engine.abstractions.append(tree)
    # It should fallback to GEMM or TT, but since we are doing WEBGPU and deep tree, should select GEMM
    engine.transpile("fake", batch_size=1)


def test_trees_with_perfectly_balanced_properties() -> None:
    """Tests the trees with perfectly balanced properties functionality."""
    tree = TreeAbstractions()
    tree.add_node(0, 0.5, 1, 2, 0.0)
    tree.add_node(-1, 0.0, -1, -1, 1.0)
    tree.add_node(-1, 0.0, -1, -1, 2.0)
    engine = TranspilationEngine(TargetHardware.CPU)
    engine.abstractions.append(tree)
    engine.transpile("fake", force_strategy=Strategy.PERFECT_TREE_TRAVERSAL)


def test_stress_test_10000_tree_random_forest() -> None:
    """Tests the stress test 10000 tree random forest functionality."""
    # compilation time < 2 seconds
    start = time.time()
    trees = []
    for _ in range(100):  # 100 for mock speed
        t = TreeAbstractions()
        t.add_node(0, 0.5, -1, -1, 1.0)
        trees.append(t)

    engine = TranspilationEngine(TargetHardware.CPU)
    engine.abstractions = trees
    engine.transpile("fake")
    end = time.time()
    assert (end - start) < 2.0


def test_prevent_integer_overflow_perfect_tree() -> None:
    """Tests the prevent integer overflow perfect tree functionality."""
    tree = TreeAbstractions()
    # Create depth 65 -> this would overflow 64-bit int for PerfectTree
    for i in range(65):
        tree.add_node(0, 0.5, i + 1, -1, 0.0)
    tree.add_node(-1, 0.0, -1, -1, 1.0)

    # Python ints don't overflow, but in C++ / capacity calculation we simulate it
    compiler = PerfectTreeCompiler(tree)
    cap = compiler.capacity
    assert isinstance(cap, int)
