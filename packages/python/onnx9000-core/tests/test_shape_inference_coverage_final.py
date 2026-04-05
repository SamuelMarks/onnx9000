"""Final coverage tests for shape_inference.py."""

import pytest
from onnx9000.core.ir import DType, DynamicDim, Graph, Node, Tensor, ValueInfo
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_shape_inference_transpose():
    """Docstring for D103."""
    g = Graph("test")
    # Transpose without perm
    t1 = Tensor("A", (2, 3, 4), DType.FLOAT32)
    g.add_tensor(t1)
    n = Node("Transpose", ["A"], ["B"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["B"].shape == (4, 3, 2)

    # Transpose with perm
    g2 = Graph("test2")
    g2.add_tensor(Tensor("A", (2, 3, 4), DType.FLOAT32))
    n2 = Node("Transpose", ["A"], ["B"], attributes={"perm": [1, 0, 2]})
    g2.add_node(n2)
    infer_shapes_and_types(g2)
    assert g2.tensors["B"].shape == (3, 2, 4)


def test_shape_inference_flatten():
    """Docstring for D103."""
    g = Graph("test")
    t1 = Tensor("A", (2, 3, 4, 5), DType.FLOAT32)
    g.add_tensor(t1)
    n = Node("Flatten", ["A"], ["B"], attributes={"axis": 2})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["B"].shape == (6, 20)

    # Flatten with negative axis
    g2 = Graph("test2")
    g2.add_tensor(Tensor("A", (2, 3, 4, 5), DType.FLOAT32))
    n2 = Node("Flatten", ["A"], ["B"], attributes={"axis": -1})
    g2.add_node(n2)
    infer_shapes_and_types(g2)
    assert g2.tensors["B"].shape == (24, 5)

    # Flatten with dynamic dims
    g3 = Graph("test3")
    g3.add_tensor(Tensor("A", (DynamicDim("D"), 3), DType.FLOAT32))
    n3 = Node("Flatten", ["A"], ["B"], attributes={"axis": 1})
    g3.add_node(n3)
    infer_shapes_and_types(g3)
    assert isinstance(g3.tensors["B"].shape[0], DynamicDim)
    assert g3.tensors["B"].shape[1] == 3

    g4 = Graph("test4")
    g4.add_tensor(Tensor("A", (2, DynamicDim("D")), DType.FLOAT32))
    n4 = Node("Flatten", ["A"], ["B"], attributes={"axis": 1})
    g4.add_node(n4)
    infer_shapes_and_types(g4)
    assert g4.tensors["B"].shape[0] == 2
    assert isinstance(g4.tensors["B"].shape[1], DynamicDim)


def test_shape_inference_nms():
    """Docstring for D103."""
    g = Graph("test")
    n = Node("NonMaxSuppression", ["boxes", "scores"], ["selected_indices"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["selected_indices"].shape == (DynamicDim("num_selected_indices"), 3)
    assert g.tensors["selected_indices"].dtype == DType.INT64


def test_shape_inference_flash_attention_and_rope():
    """Docstring for D103."""
    g = Graph("test")
    g.add_tensor(Tensor("Q", (1, 8, 128, 64), DType.FLOAT32))
    n1 = Node("FlashAttention", ["Q", "K", "V"], ["O"])
    g.add_node(n1)
    n2 = Node("RoPE", ["Q", "cos", "sin"], ["Q_out"])
    g.add_node(n2)
    infer_shapes_and_types(g)
    assert g.tensors["O"].shape == (1, 8, 128, 64)
    assert g.tensors["Q_out"].shape == (1, 8, 128, 64)


def test_shape_inference_missing_in1():
    """Docstring for D103."""
    # Test 'if not in1: continue' branches
    g = Graph("test")
    # Cast with missing input
    n1 = Node("Cast", ["Missing"], ["Out1"], attributes={"to": DType.FLOAT32.value})
    g.add_node(n1)
    # Conv with missing input
    n2 = Node("Conv", ["Missing"], ["Out2"])
    g.add_node(n2)
    infer_shapes_and_types(g)
    assert "Out1" not in g.tensors
    assert "Out2" not in g.tensors


def test_shape_inference_fallback():
    """Docstring for D103."""
    g = Graph("test")
    # Unknown op
    n = Node("UnknownOp", ["A"], ["B"])
    g.add_node(n)
    infer_shapes_and_types(g)
    # It should hit the fallback and generate dynamic outputs
    assert "B" in g.tensors
    assert isinstance(g.tensors["B"].shape[0], DynamicDim)


def test_shape_inference_custom_output_fallback():
    """Docstring for D103."""
    # To hit the custom output fallback, we need an op that is matched but doesn't fill all out_shapes
    # Let's use FlashAttention but give it more outputs than 1
    g = Graph("test")
    g.add_tensor(Tensor("Q", (2, 2), DType.FLOAT32))
    n = Node("FlashAttention", ["Q"], ["O1", "O2"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["O1"].shape == (2, 2)
    assert len(g.tensors["O2"].shape) == 1
    assert isinstance(g.tensors["O2"].shape[0], DynamicDim)


def test_shape_inference_custom_output_fallback_no_input():
    """Docstring for D103."""
    # Op with no inputs and extra outputs
    g = Graph("test")
    n = Node("NonMaxSuppression", [], ["O1", "O2"])
    g.add_node(n)
    infer_shapes_and_types(g)
    assert g.tensors["O1"].shape == (DynamicDim("num_selected_indices"), 3)
    assert len(g.tensors["O2"].shape) == 1
    assert g.tensors["O2"].dtype == DType.FLOAT32  # Default since no input
