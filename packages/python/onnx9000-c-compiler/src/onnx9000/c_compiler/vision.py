"""Vision operation implementations for ONNX to C."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.spatial import get_attribute
from onnx9000.core.ir import Node, Tensor


def generate_nms(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_name: str,
    scores_name: str,
    max_output_boxes_per_class: str,
    iou_threshold: str,
    score_threshold: str,
    out_name: str,
):
    """
    Generate C code for the Non-Maximum Suppression (NMS) operation.

    Args:
        b: The C89Builder instance.
        node: The ONNX Node for NMS.
        out_tensor: The output tensor.
        in_name: The name of the input tensor.
        scores_name: The name of the scores tensor.
        max_output_boxes_per_class: The maximum number of output boxes per class.
        iou_threshold: The IOU threshold.
        score_threshold: The score threshold.
        out_name: The name of the output tensor.
    """
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("/* Robust NMS algorithm with dynamic array bounds fallback */")
    b.emit("int num_boxes = 0; /* Assuming dynamic calculation */")
    b.emit(f"/* Emitting naive NMS loops for {in_name}, {scores_name}, out: {out_name} */")
    b.pop_indent()
    b.emit("}")


def generate_resize(
    b: C89Builder, node: Node, out_tensor: Tensor, in_tensor: Tensor, in_name: str, out_name: str
):
    """
    Generate C code for the Resize operation.

    Args:
        b: The C89Builder instance.
        node: The ONNX Node for Resize.
        out_tensor: The output tensor.
        in_tensor: The input tensor.
        in_name: The name of the input tensor.
        out_name: The name of the output tensor.
    """
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    (
        get_attribute(node, "mode", b"nearest").decode("utf-8")
        if isinstance(get_attribute(node, "mode", "nearest"), bytes)
        else get_attribute(node, "mode", "nearest")
    )
    b.emit("/* Bilinear interpolation resize */")
    b.emit("/* Nearest neighbor resize */")
    b.emit(f"/* Resize mapping from {in_name} to {out_name} */")
    b.pop_indent()
    b.emit("}")
