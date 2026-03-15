"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import infer_shapes_and_types
from onnx9000.core.exceptions import CompilationError


def test_infer_matmul():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="A", shape=(2, 3, 4), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="B", shape=(2, 4, 5), dtype=DType.FLOAT32))
    node = Node(op_type="MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    graph.add_node(node)
    infer_shapes_and_types(graph)
    assert graph.tensors["Y"].shape == (2, 3, 5)


def test_infer_gemm():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="A", shape=(4, 3), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="B", shape=(5, 4), dtype=DType.FLOAT32))
    node = Node(
        op_type="Gemm",
        inputs=["A", "B"],
        outputs=["Y"],
        attributes={"trans_a": 1, "trans_b": 1},
    )
    graph.add_node(node)
    infer_shapes_and_types(graph)
    assert graph.tensors["Y"].shape == (3, 5)


def test_infer_matmul_gemm_missing_inputs():
    """Provides semantic logic and verification."""
    graph = Graph(name="test")
    node = Node(op_type="MatMul", inputs=["A"], outputs=["Y"], attributes={})
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
    graph = Graph(name="test")
    node = Node(op_type="Gemm", inputs=["A"], outputs=["Y"], attributes={})
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
    graph = Graph(name="test")
    graph.add_tensor(Tensor(name="A", shape=(2,), dtype=DType.FLOAT32))
    graph.add_tensor(Tensor(name="B", shape=(2,), dtype=DType.FLOAT32))
    node = Node(op_type="MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    graph.add_node(node)
    with pytest.raises(CompilationError):
        infer_shapes_and_types(graph)
