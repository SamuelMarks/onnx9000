"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import infer_shapes_and_types
from onnx9000.core.exceptions import CompilationError


def test_infer_conv():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="x", shape=(1, 3, 224, 224), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="w", shape=(64, 3, 7, 7), dtype=DType.FLOAT32))
    node = Node(
        op_type="Conv",
        inputs=["x", "w"],
        outputs=["y"],
        attributes={"strides": [2, 2], "pads": [3, 3, 3, 3]},
    )
    graph.add_node(node)
    infer_shapes_and_types(graph)
    out_shape = graph.tensors["y"].shape
    assert out_shape == (1, 64, 112, 112)


def test_infer_conv_dynamic():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    graph.add_tensor(
        Tensor(name="x", shape=(1, 3, DynamicDim(-1), 224), dtype=DType.FLOAT32)
    )
    graph.add_tensor(Tensor(name="w", shape=(64, 3, 7, 7), dtype=DType.FLOAT32))
    node = Node(op_type="Conv", inputs=["x", "w"], outputs=["y"], attributes={})
    graph.add_node(node)
    infer_shapes_and_types(graph)
    out_shape = graph.tensors["y"].shape
    assert out_shape[0] == 1
    assert out_shape[1] == 64
    assert isinstance(out_shape[2], DynamicDim)
    assert out_shape[3] == 218


def test_infer_conv_missing_inputs():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    node = Node(op_type="Conv", inputs=["x"], outputs=["y"], attributes={})
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="x", shape=(1, 3, 224, 224), dtype=DType.FLOAT32))
    node = Node(op_type="Conv", inputs=["x", "w"], outputs=["y"], attributes={})
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
