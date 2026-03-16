from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.simplifier.api import simplify


def test_simplify_redundant_reshape_nodes():
    g = Graph("mock_resnet")
    g.add_node(Node("Reshape", ["input", "shape1"], ["t1"], {}, "reshape1"))
    g.add_node(Node("Reshape", ["t1", "shape2"], ["out"], {}, "reshape2"))
    g.inputs = ["input", "shape1", "shape2"]
    g.outputs = ["out"]
    g_sim = simplify(g)
    assert len(g_sim.nodes) == 1
    assert g_sim.nodes[0].op_type == "Reshape"
    assert g_sim.nodes[0].inputs == ["input", "shape2"]
    assert g_sim.nodes[0].outputs == ["out"]
