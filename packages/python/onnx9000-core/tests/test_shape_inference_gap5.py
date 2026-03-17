"""Tests the shape inference gap5 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import DynamicDim


def test_split():
    """Tests the split functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("Split", ["x"], ["y1", "y2"], {"axis": Attribute("axis", value=1)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y1"].shape) == [10, 10]
    assert list(g.tensors["y2"].shape) == [10, 10]


def test_split_with_values():
    """Tests the split with values functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_splits = Tensor("splits", (2,), DType.INT64)
    t_splits.values = [5, 15]
    g.add_tensor(t_splits)
    g.add_node(Node("Split", ["x", "splits"], ["y1", "y2"], {"axis": Attribute("axis", value=1)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y1"].shape) == [10, 5]
    assert list(g.tensors["y2"].shape) == [10, 15]


def test_tile():
    """Tests the tile functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_repeats = Tensor("repeats", (2,), DType.INT64)
    t_repeats.values = [2, 3]
    g.add_tensor(t_repeats)
    g.add_node(Node("Tile", ["x", "repeats"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [20, 60]


def test_expand():
    """Tests the expand functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 1), DType.FLOAT32))
    g.inputs.append("x")
    t_target = Tensor("shape", (2,), DType.INT64)
    t_target.values = [10, 20]
    g.add_tensor(t_target)
    g.add_node(Node("Expand", ["x", "shape"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [10, 20]


def test_expand_exception():
    """Tests the expand exception functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_target = Tensor("shape", (2,), DType.INT64)
    t_target.values = [10, 30]  # incompatible broadcast
    g.add_tensor(t_target)
    g.add_node(Node("Expand", ["x", "shape"], ["y"]))
    infer_shapes_and_types(g)
    assert "expanded_0" in str(g.tensors["y"].shape)


def test_pad():
    """Tests the pad functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_pads = Tensor("pads", (4,), DType.INT64)
    t_pads.values = [1, 2, 3, 4]
    g.add_tensor(t_pads)
    g.add_node(Node("Pad", ["x", "pads"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [14, 26]  # 10+1+3=14, 20+2+4=26
