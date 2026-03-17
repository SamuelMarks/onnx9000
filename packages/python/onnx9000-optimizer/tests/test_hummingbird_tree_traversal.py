import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.tree_traversal import TreeTraversalCompiler
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions


def test_tree_traversal_compiler():
    g = Graph(name="test_tt")
    tree = TreeAbstractions()
    tree.add_node(0, 1.5, 1, 2, 0.0)
    tree.add_node(1, 0.0, -1, -1, 10.0)
    tree.add_node(2, 0.0, -1, -1, 20.0)

    compiler = TreeTraversalCompiler(tree)
    compiler.compile(g)

    # Max depth for this tree is 2
    assert compiler.max_depth == 2

    # Check that nodes are unrolled 2 times
    op_types = [node.op_type for node in g.nodes]
    assert (
        op_types.count("Gather") > 2
    )  # feature, threshold, left, right, input_feat, values (at end)
    assert op_types.count("Less") == 4  # One for threshold, one for leaf check per depth
    assert op_types.count("Where") == 4  # One for branching, one for leaf freezing per depth
