from onnx9000.core.ir import Graph


def test_uniquify_empty():
    g = Graph("test")
    name = g._uniquify_node_name("")
    assert name == "node"
