"""Tests the shape inference gap2 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_gemm_incompatible_batch():
    """Tests the gemm incompatible batch functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (2, 3, 2), DType.FLOAT32))
    g.add_tensor(Tensor("b", (5, 2, 4), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Gemm", ["a", "b"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [5, 3, 4]


def test_gemm_1d():
    """Tests the gemm 1d functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (3,), DType.FLOAT32))
    g.add_tensor(Tensor("b", (3, 4), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Gemm", ["a", "b"], ["y"]))
    infer_shapes_and_types(g)
    assert g.tensors["y"].shape == ()


def test_cast_shape_size():
    """Tests the cast shape size functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (2, 3), DType.FLOAT32))
    g.inputs.append("x")

    g.add_node(Node("Cast", ["x"], ["y_cast"], {"to": Attribute("to", value=DType.INT32.value)}))
    g.add_node(Node("Shape", ["x"], ["y_shape"]))
    g.add_node(Node("Size", ["x"], ["y_size"]))

    infer_shapes_and_types(g)
    assert g.tensors["y_cast"].dtype == DType.INT32
    assert list(g.tensors["y_shape"].shape) == [2]
    assert g.tensors["y_size"].shape == ()
    assert g.tensors["y_size"].dtype == DType.INT64


def test_reshape_no_shape_input():
    """Tests the reshape no shape input functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (2, 3), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("Reshape", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert "y" not in g.tensors or g.tensors["y"].shape == ()
