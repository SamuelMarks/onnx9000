"""Tests the shape inference gap3 module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.symbolic import DynamicDim


def test_reshape_dynamic_vol_exception():
    """Tests the reshape dynamic vol exception functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (DynamicDim("N"), 3), DType.FLOAT32))
    t_shape = Tensor("shape", (2,), DType.INT64)
    t_shape.values = [-1, 6]
    t_shape.is_initializer = True
    g.add_tensor(t_shape)
    g.initializers.append("shape")
    g.inputs.append("x")
    g.add_node(Node("Reshape", ["x", "shape"], ["y"]))
    infer_shapes_and_types(g)
    assert g.tensors["y"].shape == (-1, 6)


def test_reshape_no_values():
    """Tests the reshape no values functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (2, 3), DType.FLOAT32))
    g.add_tensor(Tensor("shape", (3,), DType.INT64))  # not initializer
    g.inputs.extend(["x", "shape"])
    g.add_node(Node("Reshape", ["x", "shape"], ["y"]))
    infer_shapes_and_types(g)
    # should fallback to dynamic dims
    assert "dim_0" in str(g.tensors["y"].shape)


def test_global_avg_pool():
    """Tests the global avg pool functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, 32, 32), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("GlobalAveragePool", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert list(g.tensors["y"].shape) == [1, 3, 1, 1]


def test_conv_no_weights():
    """Tests the conv no weights functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1,), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("Conv", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert "C_out" in str(g.tensors["y"].shape)


def test_maxpool_channels():
    """Tests the maxpool channels functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1,), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("MaxPool", ["x"], ["y"]))
    infer_shapes_and_types(g)
    assert "C_out" in str(g.tensors["y"].shape)


def test_conv_dynamic_spatial():
    """Tests the conv dynamic spatial functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, DynamicDim("H"), 32), DType.FLOAT32))
    g.add_tensor(Tensor("w", (16, 3, 3, 3), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(
        Node(
            "Conv",
            ["x", "w"],
            ["y"],
            {
                "kernel_shape": Attribute("kernel_shape", value=[3, 3]),
                "strides": Attribute("strides", value=[1, 1]),
                "pads": Attribute("pads", value=[1, 1, 1, 1]),
                "dilations": Attribute("dilations", value=[1, 1]),
            },
        )
    )
    infer_shapes_and_types(g)
    assert "spatial_0" in str(g.tensors["y"].shape)


def test_conv_exception():
    """Tests the conv exception functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (1, 3, 32, 32), DType.FLOAT32))
    g.add_tensor(Tensor("w", (16, 3, 3, 3), DType.FLOAT32, is_initializer=True))
    g.initializers.append("w")
    g.inputs.append("x")
    g.add_node(
        Node(
            "Conv",
            ["x", "w"],
            ["y"],
            {
                "kernel_shape": Attribute("kernel_shape", value=[3]),  # intentionally short
                "strides": Attribute("strides", value=[1]),
                "pads": Attribute("pads", value=[1]),
                "dilations": Attribute("dilations", value=[1]),
            },
        )
    )
    infer_shapes_and_types(g)
    assert "spatial_0" in str(g.tensors["y"].shape)
