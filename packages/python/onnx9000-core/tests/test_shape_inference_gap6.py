"""Tests the shape inference gap6 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import DynamicDim


def test_tile_dynamic():
    """Tests the tile dynamic functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (DynamicDim("N"), 20), DType.FLOAT32))
    g.inputs.append("x")
    t_repeats = Tensor("repeats", (2,), DType.INT64)
    t_repeats.values = [2, 3]
    g.add_tensor(t_repeats)
    g.add_node(Node("Tile", ["x", "repeats"], ["y"]))
    infer_shapes_and_types(g)
    assert "tiled_0" in str(g.tensors["y"].shape)


def test_pad_dynamic():
    """Tests the pad dynamic functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (DynamicDim("N"), 20), DType.FLOAT32))
    g.inputs.append("x")
    t_pads = Tensor("pads", (4,), DType.INT64)
    t_pads.values = [1, 2, 3, 4]
    g.add_tensor(t_pads)
    g.add_node(Node("Pad", ["x", "pads"], ["y"]))
    infer_shapes_and_types(g)
    assert "padded_0" in str(g.tensors["y"].shape)


def test_topk_argmax():
    """Tests the topk argmax functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_k = Tensor("k", (1,), DType.INT64)
    t_k.values = [5]
    g.add_tensor(t_k)
    g.add_node(Node("TopK", ["x", "k"], ["y_val", "y_idx"], {"axis": Attribute("axis", value=-1)}))
    g.add_node(
        Node(
            "ArgMax",
            ["x"],
            ["y_arg"],
            {"axis": Attribute("axis", value=-1), "keepdims": Attribute("keepdims", value=0)},
        )
    )
    g.add_node(
        Node(
            "ArgMin",
            ["x"],
            ["y_arg2"],
            {"axis": Attribute("axis", value=-1), "keepdims": Attribute("keepdims", value=1)},
        )
    )
    infer_shapes_and_types(g)
    assert list(g.tensors["y_val"].shape) == [10, 5]
    assert list(g.tensors["y_arg"].shape) == [10]
    assert list(g.tensors["y_arg2"].shape) == [10, 1]


def test_topk_missing_inputs():
    """Tests the topk missing inputs functionality."""
    g = Graph("g")
    g.add_node(Node("TopK", [], ["y"]))
    g.add_node(Node("TopK", ["x"], ["y2"]))
    infer_shapes_and_types(g)


def test_if_shape_inference():
    """Tests the if shape inference functionality."""
    g = Graph("g")
    then_g = Graph("then")
    then_g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    then_g.inputs.append("x")
    then_g.add_node(Node("Relu", ["x"], ["y"]))
    then_g.outputs.append("y")
    else_g = Graph("else")
    g.add_node(
        Node(
            "If",
            ["cond"],
            ["y"],
            {
                "then_branch": Attribute("then_branch", value=then_g),
                "else_branch": Attribute("else_branch", value=else_g),
            },
        )
    )
    infer_shapes_and_types(g)


def test_loop_shape_inference():
    """Tests the loop shape inference functionality."""
    g = Graph("g")
    body_g = Graph("body")
    body_g.add_tensor(Tensor("b_in", (10, 20), DType.FLOAT32))
    body_g.inputs.extend(["iter", "cond", "b_in"])
    body_g.add_node(Node("Relu", ["b_in"], ["b_out"]))
    body_g.add_node(Node("Identity", ["cond"], ["cond_out"]))
    body_g.outputs.extend(["cond_out", "b_out", "b_out"])

    t_M = Tensor("M", (1,), DType.INT64)
    t_M.values = [5]
    g.add_tensor(t_M)
    g.inputs.append("M")

    g.add_tensor(Tensor("cond", (1,), DType.BOOL))
    g.inputs.append("cond")
    g.add_tensor(Tensor("v_in", (10, 20), DType.FLOAT32))
    g.inputs.append("v_in")

    g.add_node(
        Node(
            "Loop",
            ["M", "cond", "v_in"],
            ["v_out", "scan_out"],
            {"body": Attribute("body", value=body_g)},
        )
    )
    infer_shapes_and_types(g)
    assert list(g.tensors["scan_out"].shape) == [5, 10, 20]


def test_nonzero():
    """Tests the nonzero functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("NonZero", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert g.tensors["y"].dtype == DType.INT64
