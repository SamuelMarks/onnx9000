"""Module providing core logic and structural definitions."""


def test_compile_partial_graph() -> None:
    """Tests the test_compile_partial_graph functionality."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.toolkit.training.autograd.compiler import extract_partial_subgraph

    g = Graph(name="test")
    g.tensors["a"] = Tensor(shape=(), dtype=1, name="a")
    g.nodes.append(Node(op_type="Identity", inputs=["a"], outputs=["b"], attributes={}, name="n"))
    sub_g = extract_partial_subgraph(g, [], [])
    assert "a" in sub_g.tensors
    assert len(sub_g.nodes) == 1
