"""Tests the shape inference gap4 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import DynamicDim


def test_conv_transpose():
    """Tests the conv transpose functionality."""
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
                "kernel_shape": Attribute("kernel_shape", value=[3, 3]),
                "strides": Attribute("strides", value=[2, 2]),
                "pads": Attribute("pads", value=[1, 1, 1, 1]),
                "dilations": Attribute("dilations", value=[1, 1]),
                "output_padding": Attribute("output_padding", value=[1, 1]),
            },
        )
    )
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [1, 16, 64, 64]


def test_conv_transpose_no_weights():
    """Tests the conv transpose no weights functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, 32, 32), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("ConvTranspose", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert "C_out" in str(g.tensors["y"].shape)


def test_conv_transpose_dynamic():
    """Tests the conv transpose dynamic functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, DynamicDim("H"), 32), DType.FLOAT32))
    g.add_tensor(Tensor("w", (3, 16, 3, 3), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(Node("ConvTranspose", ["x", "w"], ["y"]))
    infer_shapes_and_types(g)
    assert "spatial_0" in str(g.tensors["y"].shape)


def test_gather():
    """Tests the gather functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.add_tensor(Tensor("indices", (5, 5), DType.INT64))
    g.inputs.extend(["x", "indices"])
    g.add_node(Node("Gather", ["x", "indices"], ["y"], {"axis": Attribute("axis", value=0)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [5, 5, 20]

    g2 = Graph("g2")
    g2.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g2.add_tensor(Tensor("indices", (5,), DType.INT64))
    g2.inputs.extend(["x", "indices"])
    g2.add_node(Node("Gather", ["x", "indices"], ["y"], {"axis": Attribute("axis", value=-1)}))
    infer_shapes_and_types(g2)
    assert list(g2.tensors["y"].shape) == [10, 5]


def test_slice():
    """Tests the slice functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))

    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [1]
    g.add_tensor(t_starts)

    t_ends = Tensor("ends", (1,), DType.INT64)
    t_ends.values = [9]
    g.add_tensor(t_ends)

    t_axes = Tensor("axes", (1,), DType.INT64)
    t_axes.values = [-1]
    g.add_tensor(t_axes)

    g.inputs.append("x")
    g.add_node(Node("Slice", ["x", "starts", "ends", "axes"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [10, 8]


def test_concat():
    """Tests the concat functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (10, 20), DType.FLOAT32))
    g.add_tensor(Tensor("b", (10, 30), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Concat", ["a", "b"], ["y"], {"axis": Attribute("axis", value=-1)}))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [10, 50]


def test_conv_transpose_short_w():
    """Tests the conv transpose short w functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, 32, 32), DType.FLOAT32))
    g.add_tensor(Tensor("w", (3, 16), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(Node("ConvTranspose", ["x", "w"], ["y"]))
    infer_shapes_and_types(g)
    assert "C_out" in str(g.tensors["y"].shape)


def test_gather_missing_inputs():
    """Tests the gather missing inputs functionality."""
    g = Graph("g")
    g.add_node(Node("Gather", ["x"], ["y"]))
    g.add_node(Node("Gather", ["x", "indices"], ["y2"]))
    infer_shapes_and_types(g)


def test_slice_missing_inputs():
    """Tests the slice missing inputs functionality."""
    g = Graph("g")
    g.add_node(Node("Slice", [], ["y"]))
    g.add_node(Node("Slice", ["x"], ["y2"]))
    infer_shapes_and_types(g)


def test_slice_default_axes_steps():
    """Tests the slice default axes steps functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [1]
    g.add_tensor(t_starts)
    g.add_tensor(Tensor("ends", (1,), DType.INT64))
    g.tensors["ends"].values = [10]
    g.add_node(Node("Slice", ["x", "starts", "ends"], ["y"]))
    infer_shapes_and_types(g)
    # axes should default to [0], steps to [1]
    assert list(g.tensors["y"].shape) == [9, 20]


def test_slice_dynamic_dim():
    """Tests the slice dynamic dim functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (DynamicDim("N"), 20), DType.FLOAT32))
    g.inputs.append("x")
    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [1]
    g.add_tensor(t_starts)
    g.add_tensor(Tensor("ends", (1,), DType.INT64))
    g.tensors["ends"].values = [10]
    g.add_node(Node("Slice", ["x", "starts", "ends"], ["y"]))
    infer_shapes_and_types(g)
    assert "sliced_0" in str(g.tensors["y"].shape)


def test_slice_negative_starts_ends():
    """Tests the slice negative starts ends functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")

    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [-5]
    g.add_tensor(t_starts)

    t_ends = Tensor("ends", (1,), DType.INT64)
    t_ends.values = [-1]
    g.add_tensor(t_ends)

    t_axes = Tensor("axes", (1,), DType.INT64)
    t_axes.values = [1]
    g.add_tensor(t_axes)

    g.add_node(Node("Slice", ["x", "starts", "ends", "axes"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [10, 4]


def test_slice_negative_step():
    """Tests the slice negative step functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")

    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [8]
    g.add_tensor(t_starts)

    t_ends = Tensor("ends", (1,), DType.INT64)
    t_ends.values = [2]
    g.add_tensor(t_ends)

    t_axes = Tensor("axes", (1,), DType.INT64)
    t_axes.values = [0]
    g.add_tensor(t_axes)

    t_steps = Tensor("steps", (1,), DType.INT64)
    t_steps.values = [-2]
    g.add_tensor(t_steps)

    g.add_node(Node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"]))
    infer_shapes_and_types(g)
    # (8 - 2 - (-2) - 1) // 2 = (8 - 2 + 2 - 1) // 2 = 7 // 2 = 3
    # Actually elements: 8, 6, 4. So 3 elements!
    assert list(g.tensors["y"].shape) == [3, 20]


def test_concat_dynamic_and_short():
    """Tests the concat dynamic and short functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("a", (10, DynamicDim("M")), DType.FLOAT32))
    # short shape to trigger axis out of bounds fallback
    g.add_tensor(Tensor("b", (30,), DType.FLOAT32))
    g.inputs.extend(["a", "b"])
    g.add_node(Node("Concat", ["a", "b"], ["y"], {"axis": Attribute("axis", value=1)}))
    infer_shapes_and_types(g)
    assert "concat_1" in str(g.tensors["y"].shape)


def test_slice_start_end_bounds():
    """Tests the slice start end bounds functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    t_starts = Tensor("starts", (1,), DType.INT64)
    t_starts.values = [-15]  # out of bounds < 0 => becomes -5 => max(0, -5) => 0
    g.add_tensor(t_starts)

    t_ends = Tensor("ends", (1,), DType.INT64)
    t_ends.values = [20]  # out of bounds > dim => min(dim, 20) => 10
    g.add_tensor(t_ends)
    g.add_node(Node("Slice", ["x", "starts", "ends"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [10, 20]


def test_concat_missing_inputs():
    """Tests the concat missing inputs functionality."""
    g = Graph("g")
    g.add_node(Node("Concat", [], ["y"]))
    g.add_node(Node("Concat", ["x"], ["y2"]))
    infer_shapes_and_types(g)
