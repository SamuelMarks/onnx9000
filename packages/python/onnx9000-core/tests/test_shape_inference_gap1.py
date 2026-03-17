"""Tests the shape inference gap1 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_promote_types():
    """Tests the promote types functionality."""
    g = Graph("g")
    # Float64
    g.add_tensor(Tensor("a", (1,), DType.FLOAT64))
    g.add_tensor(Tensor("b", (1,), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Add", ["a", "b"], ["y_f64"]))

    # Float16
    g.add_tensor(Tensor("c", (1,), DType.FLOAT16))
    g.add_tensor(Tensor("d", (1,), DType.INT8))
    g.inputs.extend(["c", "d"])
    g.add_node(Node("Add", ["c", "d"], ["y_f16"]))

    # Int64
    g.add_tensor(Tensor("e", (1,), DType.INT64))
    g.add_tensor(Tensor("f", (1,), DType.INT32))
    g.inputs.extend(["e", "f"])
    g.add_node(Node("Add", ["e", "f"], ["y_i64"]))

    # Int32
    g.add_tensor(Tensor("g_in", (1,), DType.INT32))
    g.add_tensor(Tensor("h_in", (1,), DType.INT8))
    g.inputs.extend(["g_in", "h_in"])
    g.add_node(Node("Add", ["g_in", "h_in"], ["y_i32"]))

    infer_shapes_and_types(g)
    assert g.tensors["y_f64"].dtype == DType.FLOAT64
    assert g.tensors["y_f16"].dtype == DType.FLOAT16
    assert g.tensors["y_i64"].dtype == DType.INT64
    assert g.tensors["y_i32"].dtype == DType.INT32


def test_shape_inference_error_add():
    """Tests the shape inference error add functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (2,), DType.FLOAT32))
    g.add_tensor(Tensor("b", (3,), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Add", ["a", "b"], ["y"]))
    with pytest.raises(ShapeInferenceError):
        infer_shapes_and_types(g)


def test_bool_ops():
    """Tests the bool ops functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (1,), DType.FLOAT32))
    g.add_tensor(Tensor("b", (1,), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Less", ["a", "b"], ["y"]))
    infer_shapes_and_types(g)
    assert g.tensors["y"].dtype == DType.BOOL


def test_where_op():
    """Tests the where op functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("cond", (1,), DType.BOOL))
    g.add_tensor(Tensor("a", (2, 1), DType.FLOAT32))
    g.add_tensor(Tensor("b", (1, 2), DType.FLOAT32))
    g.inputs.extend(["cond", "a", "b"])
    g.add_node(Node("Where", ["cond", "a", "b"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [2, 2]


def test_gemm_trans():
    """Tests the gemm trans functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (2, 3), DType.FLOAT32))
    g.add_tensor(Tensor("b", (4, 3), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Gemm", ["a", "b"], ["y"], {"transB": Attribute("transB", value=1)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [2, 4]

    g2 = Graph("g2")
    g2.add_tensor(Tensor("a", (3, 2), DType.FLOAT32))
    g2.add_tensor(Tensor("b", (3, 4), DType.FLOAT32))
    g2.inputs.extend(["a", "b"])
    g2.add_node(Node("Gemm", ["a", "b"], ["y"], {"transA": Attribute("transA", value=1)}))
    infer_shapes_and_types(g2)
    assert list(g2.tensors["y"].shape) == [2, 4]


def test_promote_types_fallback():
    """Tests the promote types fallback functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (1,), DType.INT8))
    g.add_tensor(Tensor("b", (1,), DType.BOOL))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Add", ["a", "b"], ["y"]))
    infer_shapes_and_types(g)
    assert g.tensors["y"].dtype == DType.INT8
