"""Tests the shape inference gap7 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_missing_inputs_all_ops():
    """Tests the missing inputs all ops functionality."""
    g = Graph("g")

    # Reshape
    g.add_node(Node("Reshape", ["x_missing", "shape_missing"], ["y1"]))  # not in1 (163)

    # Conv / MaxPool
    g.add_node(Node("Conv", [], ["y2"]))  # len < 1 (208)

    # Gather
    g.add_node(Node("Gather", ["x_missing", "i_missing"], ["y4"]))  # not in1 or not in2 (312)

    # Split
    g.add_node(Node("Split", [], ["y5"]))  # len < 1 (419)
    g.add_node(Node("Split", ["x_missing"], ["y6"]))  # not in1 (422)

    # Tile / Expand
    g.add_node(Node("Tile", [], ["y7"]))  # len < 2 (449)
    g.add_node(Node("Tile", ["x_missing", "repeats_missing"], ["y8"]))  # not in1 (455)

    # Pad
    g.add_node(Node("Pad", ["x_missing"], ["y9"]))  # not in1 (487)

    # NonZero
    g.add_node(Node("NonZero", [], ["y10"]))  # len < 1 (650)
    g.add_node(Node("NonZero", ["x_missing"], ["y11"]))  # not in1 (653)

    infer_shapes_and_types(g)


def test_tile_repeats_no_values():
    """Tests the tile repeats no values functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    g.add_tensor(Tensor("repeats", (2,), DType.INT64))  # no values
    g.inputs.append("repeats")
    g.add_node(Node("Tile", ["x", "repeats"], ["y"]))
    infer_shapes_and_types(g)
    # This hits 458: if shape_t and hasattr(shape_t, "values") and shape_t.values is not None:
    # repeats remains []


def test_conv_no_spatial_dims():
    """Tests the conv no spatial dims functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3), DType.FLOAT32))  # len(in_shape) = 2
    g.inputs.append("x")
    g.add_tensor(Tensor("w", (16, 3), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.add_node(Node("Conv", ["x", "w"], ["y"]))
    infer_shapes_and_types(g)
    # This might hit 260, 263?


def test_split_no_splits():
    """Tests the split no splits functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    g.add_tensor(Tensor("splits", (2,), DType.INT64))  # no values
    g.inputs.append("splits")
    g.add_node(Node("Split", ["x", "splits"], ["y1", "y2"]))
    infer_shapes_and_types(g)


def test_loop_m_no_values():
    """Tests the loop m no values functionality."""
    g = Graph("g")
    body_g = Graph("body")
    body_g.add_tensor(Tensor("b_in", (10, 20), DType.FLOAT32))
    body_g.inputs.extend(["iter", "cond", "b_in"])
    body_g.add_node(Node("Relu", ["b_in"], ["b_out"]))
    body_g.add_node(Node("Identity", ["cond"], ["cond_out"]))
    body_g.outputs.extend(["cond_out", "b_out", "b_out"])

    t_M = Tensor("M", (1,), DType.INT64)  # no values
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
    # This hits 642-643 (fallback to loop_iters)
