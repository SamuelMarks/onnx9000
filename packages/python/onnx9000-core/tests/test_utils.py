"""Module docstring."""


def test_topological_sort():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.utils import CyclicDependencyError, topological_sort

    g = Graph("test")
    n1 = Node("Add", inputs=[], outputs=["a"])
    n2 = Node("Add", inputs=["a"], outputs=["b"])
    g.nodes.extend([n2, n1])

    sorted_nodes = topological_sort(g)
    assert sorted_nodes[0] == n1
    assert sorted_nodes[1] == n2


def test_topological_sort_cyclic():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.utils import CyclicDependencyError, topological_sort

    g = Graph("test")
    n1 = Node("Add", inputs=["b"], outputs=["a"])
    n2 = Node("Add", inputs=["a"], outputs=["b"])
    g.nodes.extend([n2, n1])

    with pytest.raises(CyclicDependencyError):
        topological_sort(g)


def test_topological_sort_already_visited():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node
    from onnx9000.core.utils import topological_sort

    g = Graph("test")
    n1 = Node("Add", inputs=[], outputs=["a"])
    n2 = Node("Add", inputs=["a"], outputs=["b"])
    n3 = Node("Add", inputs=["a"], outputs=["c"])
    g.nodes.extend([n2, n3, n1])

    sorted_nodes = topological_sort(g)
    assert sorted_nodes[0] == n1
