"""Tests the shape inference gap8 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_conv_transpose_exception():
    """Tests the conv transpose exception functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, 32, 32), DType.FLOAT32))
    g.add_tensor(Tensor("w", (3, 16, 3, 3), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(
        Node(
            "ConvTranspose",
            ["x", "w"],
            ["y"],
            {
                "kernel_shape": Attribute("kernel_shape", value=[3]),  # intentionally short
                "strides": Attribute("strides", value=[2]),
                "pads": Attribute("pads", value=[1, 1]),
                "dilations": Attribute("dilations", value=[1]),
            },
        )
    )
    infer_shapes_and_types(g)
    assert "spatial_0" in str(g.tensors["y"].shape)


def test_split_negative_axis():
    """Tests the split negative axis functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("Split", ["x"], ["y1", "y2"], {"axis": Attribute("axis", value=-1)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y1"].shape) == [10, 10]


def test_missing_inputs_again():
    """Tests the missing inputs again functionality."""
    # If the nodes have outputs that are NOT appended to graph.outputs,
    # and inputs NOT in graph.inputs, maybe they are pruned?
    # Let's append to graph.outputs
    g = Graph("g")

    g.add_node(Node("Gather", ["x_missing", "i_missing"], ["y_gather"]))
    g.outputs.append("y_gather")

    g.add_node(Node("Tile", ["x_missing"], ["y_tile"]))  # len < 2
    g.outputs.append("y_tile")

    g.add_node(Node("Pad", ["x_pad_missing"], ["y_pad"]))
    g.outputs.append("y_pad")

    infer_shapes_and_types(g)


def test_loop_m_scalar():
    """Tests the loop m scalar functionality."""
    g = Graph("g")
    body_g = Graph("body")
    body_g.add_tensor(Tensor("b_in", (10, 20), DType.FLOAT32))
    body_g.inputs.extend(["iter", "cond", "b_in"])
    body_g.add_node(Node("Relu", ["b_in"], ["b_out"]))
    body_g.add_node(Node("Identity", ["cond"], ["cond_out"]))
    body_g.outputs.extend(["cond_out", "b_out", "b_out"])

    g.add_tensor(Tensor("M", (), DType.INT64))  # scalar shape!
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
    assert "loop_iters" in str(g.tensors["scan_out"].shape)
