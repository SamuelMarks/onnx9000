"""Module providing core logic and structural definitions."""


def test_compile_partial_graph():
    """Provides semantic functionality and verification."""
    from onnx9000.training.autograd.compiler import extract_partial_subgraph
    from onnx9000.core.ir import Graph, Tensor, Node

    g = Graph(name="test")
    g.tensors["a"] = Tensor(shape=(), dtype=1, name="a")
    g.nodes.append(
        Node(op_type="Identity", inputs=["a"], outputs=["b"], attributes={}, name="n")
    )
    sub_g = extract_partial_subgraph(g, [], [])
    assert "a" in sub_g.tensors
    assert len(sub_g.nodes) == 1
