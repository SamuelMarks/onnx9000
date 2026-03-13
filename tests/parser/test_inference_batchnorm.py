"""Module docstring."""

import pytest
from onnx9000.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.dtypes import DType
from onnx9000.parser.inference import infer_shapes_and_types
from onnx9000.exceptions import CompilationError


def test_infer_batchnorm():
    """Test function docstring."""
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="x", shape=(2, 3, 4, 5), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="scale", shape=(3,), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="b", shape=(3,), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="mean", shape=(3,), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="var", shape=(3,), dtype=DType.FLOAT32))
    node = Node(
        op_type="BatchNormalization",
        inputs=["x", "scale", "b", "mean", "var"],
        outputs=["y", "rm", "rv", "sm", "sv"],
        attributes={},
    )
    graph.add_node(node)

    infer_shapes_and_types(graph)
    assert graph.tensors["y"].shape == (2, 3, 4, 5)
    assert graph.tensors["rm"].shape == (3,)
    assert graph.tensors["rv"].shape == (3,)


def test_infer_batchnorm_missing():
    """Test function docstring."""
    graph = Graph(name="test")
    node = Node(
        op_type="BatchNormalization",
        inputs=["x", "scale"],
        outputs=["y"],
        attributes={},
    )
    graph.add_node(node)

    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)

    graph = Graph(name="test")
    node = Node(
        op_type="BatchNormalization",
        inputs=["x", "scale", "b", "mean", "var"],
        outputs=["y"],
        attributes={},
    )
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
