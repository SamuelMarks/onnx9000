import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.tree_traversal import (
    compile_forest_tree_traversal,
    handle_categorical_traversal,
    handle_missing_value_traversal,
    flatten_multi_class_traversal,
    test_gather_latency_wasm,
)


def test_tree_traversal_stubs():
    g = Graph("g")
    compile_forest_tree_traversal(g, [], 1)
    handle_categorical_traversal(g)
    handle_missing_value_traversal(g)
    flatten_multi_class_traversal(g)
    test_gather_latency_wasm()


from onnx9000.optimizer.hummingbird.tree_traversal import TreeTraversalCompiler
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions


def test_tree_traversal_compiler():
    g = Graph("g")
    t = TreeAbstractions()
    t.add_node(1, 0.5, 1, 2, 0.0)
    t.add_node(-1, 0.0, -1, -1, 1.0)
    t.add_node(-1, 0.0, -1, -1, -1.0)
    c = TreeTraversalCompiler(t, 1)
    c.compile(g)
    assert len(g.nodes) > 0
