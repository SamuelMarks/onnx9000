"""Tests the shape inference gap9 module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_split_dynamic():
    """Tests the split dynamic functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", (10, 20), DType.FLOAT32))
    g.inputs.append("x")
    # Split axis 1 (20) into 3 pieces => 20 % 3 != 0 => fallback to DynamicDim
    g.add_node(Node("Split", ["x"], ["y1", "y2", "y3"], {"axis": Attribute("axis", value=1)}))
    infer_shapes_and_types(g)
    assert "split_1" in str(g.tensors["y1"].shape)


def test_loop_missing_body_out():
    """Tests the loop missing body out functionality."""
    g = Graph("g")
    body_g = Graph("body")
    # body graph inputs
    body_g.add_tensor(Tensor("b_in", (10, 20), DType.FLOAT32))
    body_g.inputs.extend(["iter", "cond", "b_in"])
    body_g.add_node(Node("Relu", ["b_in"], ["b_out"]))
    body_g.add_node(Node("Identity", ["cond"], ["cond_out"]))
    # Add a missing output name
    body_g.outputs.extend(["cond_out", "b_out", "missing_out"])

    g.add_tensor(Tensor("M", (), DType.INT64))
    g.inputs.append("M")

    g.add_tensor(Tensor("cond", (1,), DType.BOOL))
    g.inputs.append("cond")
    g.add_tensor(Tensor("v_in", (10, 20), DType.FLOAT32))
    g.inputs.append("v_in")

    g.add_node(
        Node(
            "Loop",
            ["M", "cond", "v_in"],
            ["v_out", "scan_out1", "scan_out2"],
            {"body": Attribute("body", value=body_g)},
        )
    )
    infer_shapes_and_types(g)
    # scan_out2 corresponds to missing_out
    assert g.tensors["scan_out2"].shape == ()
    assert g.tensors["scan_out2"].dtype == DType.FLOAT32
