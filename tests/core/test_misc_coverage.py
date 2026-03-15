"""Module providing core logic and structural definitions."""


def test_memory_planning_reused_output():
    """Provides semantic functionality and verification."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.parser.memory import plan_memory

    g = Graph(name="test")
    g.inputs.append("x")
    g.tensors["x"] = Tensor("x", shape=(), dtype=1)
    g.nodes.append(
        Node(op_type="Identity", inputs=["x"], outputs=["x"], attributes={}, name="n1")
    )
    plan_memory(g)


def test_deform_conv_validation():
    import pytest
    import onnx9000.core.ops as ops
    from onnx9000.frontends.frontend.tensor import Tensor

    x = Tensor(name="x")
    w = Tensor(name="w")
    offset = Tensor(name="offset")
    mask = Tensor(name="mask")
    with pytest.raises(ValueError, match="Mask provided without Bias"):
        ops.deform_conv(x, w, offset, mask=mask)


def test_non_max_suppression_validation():
    import pytest
    import onnx9000.core.ops as ops
    from onnx9000.frontends.frontend.tensor import Tensor

    boxes = Tensor(name="boxes")
    scores = Tensor(name="scores")
    iou_threshold = Tensor(name="iou_threshold")
    score_threshold = Tensor(name="score_threshold")
    with pytest.raises(ValueError, match="ONNX sequential inputs rule violation"):
        ops.non_max_suppression(boxes, scores, iou_threshold=iou_threshold)
    with pytest.raises(ValueError, match="ONNX sequential inputs rule violation"):
        ops.non_max_suppression(
            boxes,
            scores,
            max_output_boxes_per_class=Tensor(name="max"),
            score_threshold=score_threshold,
        )
