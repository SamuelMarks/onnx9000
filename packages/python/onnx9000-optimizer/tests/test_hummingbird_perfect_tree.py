from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions
from onnx9000.optimizer.hummingbird.perfect_tree import PerfectTreeCompiler


def test_perfect_tree_compiler() -> None:
    g = Graph(name="test_pt")
    tree = TreeAbstractions()
    tree.add_node(0, 1.5, 1, 2, 0.0)
    tree.add_node(1, 0.0, -1, -1, 10.0)
    tree.add_node(2, 0.0, -1, -1, 20.0)

    compiler = PerfectTreeCompiler(tree)
    compiler.compile(g)

    # Max depth for this tree is 2
    assert compiler.max_depth == 2
    assert compiler.capacity == 3

    # Tensors generated
    assert "pt_feat" in g.tensors
    assert "pt_thresh" in g.tensors
    assert "pt_val" in g.tensors

    op_types = [node.op_type for node in g.nodes]
    # No left/right Gather ops because it's computed purely mathematically
    assert op_types.count("Mul") == 2
    assert op_types.count("Add") == 2
    assert op_types.count("Sub") == 2
