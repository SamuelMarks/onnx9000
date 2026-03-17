import pytest
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions
from onnx9000.optimizer.hummingbird.analysis import (
    analyze_tree_depth,
    analyze_leaf_distribution,
    flatten_ensemble,
    cast_parameters,
)


def test_analyze_tree_depth():
    tree = TreeAbstractions()
    tree.add_node(0, 1.5, 1, 2, 0.0)
    tree.add_node(1, 0.0, -1, -1, 10.0)
    tree.add_node(2, 0.0, -1, -1, 20.0)

    depths = analyze_tree_depth(tree)
    assert depths["min"] == 2
    assert depths["max"] == 2
    assert depths["mean"] == 2.0


def test_analyze_leaf_distribution():
    tree = TreeAbstractions()
    tree.add_node(0, 1.5, 1, 2, 0.0)
    tree.add_node(-1, 0.0, -1, -1, 10.0)
    tree.add_node(-1, 0.0, -1, -1, 20.0)

    dist = analyze_leaf_distribution(tree)
    assert dist[10.0] == 1
    assert dist[20.0] == 1


def test_flatten_ensemble():
    tree1 = TreeAbstractions()
    tree1.add_node(0, 1.5, -1, -1, 1.0)

    tree2 = TreeAbstractions()
    tree2.add_node(1, 2.5, -1, -1, 2.0)

    flattened = flatten_ensemble([tree1, tree2])
    assert len(flattened.features) == 2
    assert flattened.values[0] == 1.0
    assert flattened.values[1] == 2.0


def test_cast_parameters():
    tree = TreeAbstractions()
    tree.add_node(0, 1.123456789123456789, -1, -1, 2.123456789123456789)
    casted = cast_parameters(tree)
    # They should be truncated to float32
    assert casted.thresholds[0] != 1.123456789123456789
    assert casted.values[0] != 2.123456789123456789


def test_analysis_empty():
    from onnx9000.optimizer.hummingbird.analysis import (
        analyze_tree_depth,
        analyze_leaf_distribution,
    )
    from onnx9000.optimizer.hummingbird.engine import TreeAbstractions

    ab = TreeAbstractions()
    assert analyze_tree_depth(ab) == {"min": 0, "max": 0, "mean": 0}
    assert analyze_leaf_distribution(ab) == {}


def test_leaf_distribution_empty():
    from onnx9000.optimizer.hummingbird.analysis import analyze_leaf_distribution
    from onnx9000.optimizer.hummingbird.engine import TreeAbstractions

    ab = TreeAbstractions()
    ab.add_node(0, 0.0, -1, -1, 1.0)
    ab.add_node(0, 0.0, -1, -1, 1.0)
    dist = analyze_leaf_distribution(ab)
    assert dist[1.0] == 2
