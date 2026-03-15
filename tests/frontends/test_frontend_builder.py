import pytest
from onnx9000.frontends.frontend.builder import (
    GraphBuilder,
    Tracing,
    get_active_builder,
)
from onnx9000.frontends.frontend.tensor import Node


def test_graph_builder():
    gb = GraphBuilder("my_graph")
    assert gb.name == "my_graph"
    assert gb.nodes == []
    assert gb.inputs == []
    assert gb.outputs == []
    assert gb.parameters == []
    n = Node("Relu", [], [], {})
    gb.add_node(n)
    assert len(gb.nodes) == 1


def test_tracing_context():
    assert get_active_builder() is None
    with Tracing() as gb1:
        assert get_active_builder() is gb1
        with Tracing(GraphBuilder("inner")) as gb2:
            assert get_active_builder() is gb2
            assert gb2.name == "inner"
        assert get_active_builder() is gb1
    assert get_active_builder() is None
