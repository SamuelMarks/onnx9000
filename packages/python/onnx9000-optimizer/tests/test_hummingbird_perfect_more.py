import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions
from onnx9000.optimizer.hummingbird.perfect_tree import PerfectTreeCompiler


def test_perfect_tree_converter():
    g = Graph("g")
    t = TreeAbstractions()
    t.add_node(1, 0.5, 1, 2, 0.0)
    t.add_node(-1, 0.0, -1, -1, 1.0)
    t.add_node(-1, 0.0, -1, -1, -1.0)
    c = PerfectTreeCompiler(t, batch_size=1)
    c._detect_and_trim_branches()
    c._pad_to_perfect_tree()
    c.compile(g)
    assert len(g.nodes) > 0


from onnx9000.optimizer.hummingbird.perfect_tree import (
    handle_perfect_multi_output,
    map_categorical_perfect,
)


def test_perfect_tree_stubs_more():
    g = Graph("g")
    handle_perfect_multi_output(g)
    map_categorical_perfect(g)
