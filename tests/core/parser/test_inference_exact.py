"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.inference import (
    infer_binary,
    infer_reshape,
    infer_conv,
    infer_batchnorm,
    infer_slice,
    infer_deform_conv,
    infer_eye_like,
    infer_mel_weight_matrix,
    infer_multinomial,
    infer_topk,
    infer_string_split,
    infer_shapes_and_types,
    _INFERENCE_RULES,
)


def test_inference_binary_dynamic_dim_and_where_cond():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(DynamicDim(-1), 1), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(1, 5), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("C", shape=(DynamicDim(-1), 1, 5), dtype=DType.BOOL))
    node1 = Node("Add", inputs=["A", "B"], outputs=["out1"], attributes={})
    infer_binary(node1, g)
    assert g.tensors["out1"].shape == (DynamicDim(-1), 5)
    node2 = Node("Where", inputs=["C", "A", "B"], outputs=["out2"], attributes={})
    infer_binary(node2, g)
    assert g.tensors["out2"].shape == (DynamicDim(-1), DynamicDim(-1), 5)


def test_inference_reshape_neg_idx():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(2, 3, 4), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("target", shape=(2,), dtype=DType.INT64))
    import numpy as np

    g.tensors["target"].data = np.array([6, -1])
    node = Node("Reshape", inputs=["X", "target"], outputs=["out"], attributes={})
    infer_reshape(node, g)
    assert g.tensors["out"].shape == (6, 4)


def test_inference_conv_dynamic_dim():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(
        Tensor("X", shape=(1, 3, DynamicDim(-1), DynamicDim(-1)), dtype=DType.FLOAT32)
    )
    g.add_tensor(Tensor("W", shape=(2, 3, 3, 3), dtype=DType.FLOAT32))
    node = Node("Conv", inputs=["X", "W"], outputs=["out"], attributes={})
    infer_conv(node, g)
    assert isinstance(g.tensors["out"].shape[2], DynamicDim)


def test_inference_batchnorm_multi_output():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(1, 3, 5, 5), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("scale", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("mean", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("var", shape=(3,), dtype=DType.FLOAT32))
    node = Node(
        "BatchNormalization",
        inputs=["X", "scale", "B", "mean", "var"],
        outputs=["Y", "mean_out", "var_out"],
        attributes={},
    )
    infer_batchnorm(node, g)
    assert g.tensors["Y"].shape == (1, 3, 5, 5)
    assert g.tensors["mean_out"].shape == (3,)
    assert g.tensors["var_out"].shape == (3,)


def test_inference_slice_negative_and_out_of_bounds():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("starts", shape=(2,), dtype=DType.INT64))
    g.add_tensor(Tensor("ends", shape=(2,), dtype=DType.INT64))
    g.add_tensor(Tensor("axes", shape=(2,), dtype=DType.INT64))
    g.add_tensor(Tensor("steps", shape=(2,), dtype=DType.INT64))
    import numpy as np

    g.tensors["starts"].data = np.array([-5, -25])
    g.tensors["ends"].data = np.array([15, -1])
    g.tensors["axes"].data = np.array([0, 1])
    g.tensors["steps"].data = np.array([1, 1])
    node = Node(
        "Slice",
        inputs=["X", "starts", "ends", "axes", "steps"],
        outputs=["out"],
        attributes={},
    )
    infer_slice(node, g)
    assert g.tensors["out"].shape == (5, 19)


def test_inference_deform_conv():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(1, 3, 10, 10), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(2, 3, 3, 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("offset", shape=(1, 18, 10, 10), dtype=DType.FLOAT32))
    node = Node(
        "DeformConv", inputs=["X", "W", "offset"], outputs=["out"], attributes={}
    )
    infer_deform_conv(node, g)
    assert g.tensors["out"].shape == (1, 2, 1, 1)


def test_inference_eye_like_int_dtype():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(3, 3), dtype=DType.FLOAT32))
    node = Node("EyeLike", inputs=["X"], outputs=["out"], attributes={"dtype": 6})
    infer_eye_like(node, g)
    assert g.tensors["out"].dtype == DType.INT32


def test_inference_mel_weight_matrix_int_dtype():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(1, 1), dtype=DType.FLOAT32))
    node = Node(
        "MelWeightMatrix",
        inputs=["X"],
        outputs=["out"],
        attributes={"output_datatype": 6},
    )
    infer_mel_weight_matrix(node, g)
    assert g.tensors["out"].dtype == DType.INT32


def test_inference_multinomial_int_dtype():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(2, 3), dtype=DType.FLOAT32))
    node = Node(
        "Multinomial",
        inputs=["X"],
        outputs=["out"],
        attributes={"dtype": 6, "sample_size": 5},
    )
    infer_multinomial(node, g)
    assert g.tensors["out"].dtype == DType.INT32


def test_inference_topk():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(2, 3, 4), dtype=DType.FLOAT32))
    node = Node(
        "TopK",
        inputs=["X", "K"],
        outputs=["Values", "Indices"],
        attributes={"axis": -1},
    )
    infer_topk(node, g)
    assert g.tensors["Values"].shape == (2, 3, 1)
    assert g.tensors["Indices"].shape == (2, 3, 1)


def test_inference_shapes_and_types_fallback():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    node = Node("MyUnknownOp", inputs=[], outputs=["Y"], attributes={})
    g.nodes.append(node)
    infer_shapes_and_types(g)
    assert g.tensors["Y"].dtype == DType.FLOAT32
    assert isinstance(g.tensors["Y"].shape[0], DynamicDim)


def test_inference_string_split():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(2,), dtype=DType.STRING))
    node = Node("StringSplit", inputs=["X"], outputs=["Y", "Z"], attributes={})
    infer_string_split(node, g)
    assert g.tensors["Z"].dtype == DType.INT64
    assert g.tensors["Z"].shape == (1,)
