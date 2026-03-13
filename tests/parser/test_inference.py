"""Module docstring."""

import pytest
from onnx9000.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.dtypes import DType
from onnx9000.parser.inference import infer_shapes_and_types
from onnx9000.exceptions import CompilationError


def test_infer_unary():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Abs", inputs=["in"], outputs=["out"], attributes={})
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert out_tensor.shape == (2, 3)
    assert out_tensor.dtype == DType.FLOAT32


def test_infer_unary_missing_input():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    # intentionally missing input tensor
    node = Node(op_type="Abs", inputs=["in"], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(CompilationError, match="Input tensor in not found"):
        infer_shapes_and_types(graph)


def test_infer_unary_invalid_args():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    node = Node(op_type="Abs", inputs=[], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(CompilationError, match="Abs expects 1 input and 1 output"):
        infer_shapes_and_types(graph)


def test_infer_binary():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in1", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="in2", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Add", inputs=["in1", "in2"], outputs=["out"], attributes={})
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert out_tensor.shape == (2, 3)
    assert out_tensor.dtype == DType.FLOAT32


def test_infer_binary_logical():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in1", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="in2", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Equal", inputs=["in1", "in2"], outputs=["out"], attributes={})
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert out_tensor.shape == (2, 3)
    assert out_tensor.dtype == DType.BOOL


def test_infer_binary_missing_args():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in1", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Add", inputs=["in1"], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(
        CompilationError, match="Add expects at least 2 inputs and 1 output"
    ):
        infer_shapes_and_types(graph)


def test_infer_binary_missing_tensors():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    node = Node(op_type="Add", inputs=["in1", "in2"], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(CompilationError, match="Missing inputs for binary op"):
        infer_shapes_and_types(graph)


def test_fallback_inference():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="UnknownOp", inputs=["in"], outputs=["out"], attributes={})
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert isinstance(out_tensor.shape[0], DynamicDim)
    assert out_tensor.shape[0].value == -1
    assert out_tensor.dtype == DType.FLOAT32


def test_infer_binary_incompatible_shapes():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in1", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="in2", shape=(2, 4), dtype=DType.FLOAT32))
    node = Node(op_type="Add", inputs=["in1", "in2"], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(CompilationError, match="Incompatible shapes for broadcast"):
        infer_shapes_and_types(graph)


def test_infer_binary_dynamic_shapes():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="in1", shape=(DynamicDim(-1), 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="in2", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Add", inputs=["in1", "in2"], outputs=["out"], attributes={})
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert isinstance(out_tensor.shape[0], DynamicDim)
    assert out_tensor.shape[1] == 3


def test_infer_where_broadcast():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="cond", shape=(1, 3), dtype=DType.BOOL))
    graph.add_tensor(Tensor(name="x", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="y", shape=(2, 1), dtype=DType.FLOAT32))
    node = Node(
        op_type="Where", inputs=["cond", "x", "y"], outputs=["out"], attributes={}
    )
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert out_tensor.shape == (2, 3)


def test_infer_where_dynamic_cond():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="cond", shape=(DynamicDim(-1), 3), dtype=DType.BOOL))
    graph.add_tensor(Tensor(name="x", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="y", shape=(2, 1), dtype=DType.FLOAT32))
    node = Node(
        op_type="Where", inputs=["cond", "x", "y"], outputs=["out"], attributes={}
    )
    graph.add_node(node)

    infer_shapes_and_types(graph)
    out_tensor = graph.tensors["out"]
    assert isinstance(out_tensor.shape[0], DynamicDim)


def test_infer_where_incompatible_cond():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="cond", shape=(4, 3), dtype=DType.BOOL))
    graph.add_tensor(Tensor(name="x", shape=(2, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="y", shape=(2, 1), dtype=DType.FLOAT32))
    node = Node(
        op_type="Where", inputs=["cond", "x", "y"], outputs=["out"], attributes={}
    )
    graph.add_node(node)

    with pytest.raises(
        CompilationError, match="Incompatible condition shape for broadcast in Where"
    ):
        infer_shapes_and_types(graph)


def test_infer_where_missing_inputs():
    """Test function docstring."""
    graph = Graph(name="test_graph")
    graph.add_tensor(Tensor(name="cond", shape=(4, 3), dtype=DType.BOOL))
    graph.add_tensor(Tensor(name="x", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(op_type="Where", inputs=["cond", "x"], outputs=["out"], attributes={})
    graph.add_node(node)

    with pytest.raises(CompilationError, match="Missing inputs for Where op."):
        infer_shapes_and_types(graph)
