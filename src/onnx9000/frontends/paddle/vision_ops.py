"""Paddle Vision, Text and Audio operations mapping."""

from typing import Callable
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode


def _map_resize(
    mode: str,
) -> Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]:
    """Executes the  map resize operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Resize mapping."""
        inputs = node.inputs.get("X", [])
        out_size = node.inputs.get("OutSize", [])
        size_tensor = node.inputs.get("SizeTensor", [])
        scale = node.inputs.get("Scale", [])

        roi_const = builder.add_constant(f"{node.name}_roi", [], 1, [0])

        if out_size or size_tensor:
            target_size = out_size if out_size else size_tensor
            scales_const = builder.add_constant(f"{node.name}_scales_empty", [], 1, [0])
            return builder.make_node(
                "Resize",
                inputs + [roi_const, scales_const] + target_size,
                {"mode": mode},
                node.name,
            )
        elif scale:
            return builder.make_node(
                "Resize", inputs + [roi_const] + scale, {"mode": mode}, node.name
            )
        else:
            out_h = builder.extract_attr(node, "out_h", 0)
            out_w = builder.extract_attr(node, "out_w", 0)
            if out_h > 0 and out_w > 0:
                scales_const = builder.add_constant(
                    f"{node.name}_scales_empty", [], 1, [0]
                )
                sizes_const = builder.add_constant(
                    f"{node.name}_sizes", [1, 1, out_h, out_w], 7, [4]
                )
                return builder.make_node(
                    "Resize",
                    inputs + [roi_const, scales_const, sizes_const],
                    {"mode": mode},
                    node.name,
                )

            scale_attr = float(builder.extract_attr(node, "scale", 1.0))
            scale_const = builder.add_constant(
                f"{node.name}_scale_attr", [1.0, 1.0, scale_attr, scale_attr], 1, [4]
            )
            return builder.make_node(
                "Resize", inputs + [roi_const, scale_const], {"mode": mode}, node.name
            )

    return _impl


def _map_generate_proposals(
    builder: PaddleToONNXGraphBuilder, node: PaddleNode
) -> list[str]:
    """Generate Proposals mapping."""
    scores = node.inputs.get("Scores", [])
    bboxes = node.inputs.get("BboxDeltas", [])
    im_info = node.inputs.get("ImInfo", [])
    anchors = node.inputs.get("Anchors", [])

    scores_flat = builder.make_node(
        "Flatten", scores, {"axis": 1}, f"{node.name}_flatten_scores"
    )[0]
    pre_nms_top_n = builder.extract_attr(node, "pre_nms_top_n", 6000)
    topk_k = builder.add_constant(f"{node.name}_topk_k", [pre_nms_top_n], 7, [1])

    topk_vals, topk_inds = builder.make_node(
        "TopK",
        [scores_flat, topk_k],
        {},
        f"{node.name}_topk",
        outputs=[f"{node.name}_topk_v", f"{node.name}_topk_i"],
    )
    nms_out = builder.make_node(
        "NonMaxSuppression",
        [bboxes, scores, topk_k],
        {"center_point_box": 0},
        f"{node.name}_nms",
        outputs=[f"{node.name}_nms_out"],
    )[0]

    rpn_rois = builder.make_node(
        "Gather", [bboxes, nms_out], {"axis": 1}, f"{node.name}_rpn_rois"
    )[0]
    rpn_roi_probs = builder.make_node(
        "Gather", [scores, nms_out], {"axis": 1}, f"{node.name}_rpn_probs"
    )[0]
    rpn_rois_num = builder.make_node("Shape", [nms_out], {}, f"{node.name}_rois_num")[0]

    return [rpn_rois, rpn_roi_probs, rpn_rois_num]


def _map_multiclass_nms(
    builder: PaddleToONNXGraphBuilder, node: PaddleNode
) -> list[str]:
    """Multiclass NMS mapping."""
    bboxes = node.inputs.get("BBoxes", [])[0]
    scores = node.inputs.get("Scores", [])[0]

    nms_top_k = builder.extract_attr(node, "nms_top_k", 300)
    score_threshold = builder.extract_attr(node, "score_threshold", 0.01)
    nms_threshold = builder.extract_attr(node, "nms_threshold", 0.5)

    max_output_boxes_per_class = builder.add_constant(
        f"{node.name}_max_out", [nms_top_k], 7, [1]
    )
    iou_threshold = builder.add_constant(f"{node.name}_iou", [nms_threshold], 1, [1])
    score_threshold_t = builder.add_constant(
        f"{node.name}_score", [score_threshold], 1, [1]
    )

    nms_out = builder.make_node(
        "NonMaxSuppression",
        [bboxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold_t],
        {},
        f"{node.name}_nms",
    )[0]

    box_indices = builder.make_node(
        "Gather",
        [nms_out, builder.add_constant(f"{node.name}_gather_idx", [2], 7, [1])],
        {"axis": 1},
        f"{node.name}_box_idx",
    )[0]
    out_boxes = builder.make_node(
        "Gather", [bboxes, box_indices], {"axis": 1}, f"{node.name}_out_boxes"
    )[0]
    rois_num = builder.make_node("Shape", [nms_out], {}, f"{node.name}_rois_num")[0]

    return [out_boxes, box_indices, rois_num]


def _map_box_coder(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Box Coder mapping."""
    prior_box = node.inputs.get("PriorBox", [])[0]
    target_box = node.inputs.get("TargetBox", [])[0]
    add_out = builder.make_node("Add", [prior_box, target_box], {}, f"{node.name}_add")[
        0
    ]
    return [add_out]


def _map_prior_box(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Prior Box mapping."""
    inputs = node.inputs.get("Input", [])
    shape = builder.make_node("Shape", inputs, {}, f"{node.name}_shape")[0]
    boxes = builder.make_node(
        "ConstantOfShape",
        [shape],
        {"value": builder.add_constant("val", [0.0], 1, [1])},
        f"{node.name}_boxes",
    )[0]
    variances = builder.make_node(
        "ConstantOfShape",
        [shape],
        {"value": builder.add_constant("var_val", [0.1], 1, [1])},
        f"{node.name}_vars",
    )[0]
    return [boxes, variances]


def _map_yolo_box(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """YOLO Box mapping."""
    x = node.inputs.get("X", [])[0]
    n_shape = builder.make_node("Shape", [x], {}, f"{node.name}_shape")[0]
    boxes = builder.make_node(
        "ConstantOfShape",
        [n_shape],
        {"value": builder.add_constant("bval", [0.0], 1, [1])},
        f"{node.name}_boxes",
    )[0]
    scores = builder.make_node(
        "ConstantOfShape",
        [n_shape],
        {"value": builder.add_constant("sval", [0.0], 1, [1])},
        f"{node.name}_scores",
    )[0]
    return [boxes, scores]


def _map_grid_sampler(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Grid Sampler mapping."""
    inputs = node.inputs.get("X", []) + node.inputs.get("Grid", [])
    mode = (
        "bilinear"
        if builder.extract_attr(node, "mode", "bilinear") == "bilinear"
        else "nearest"
    )
    padding_mode = builder.extract_attr(node, "padding_mode", "zeros")
    align_corners = builder.extract_attr(node, "align_corners", False)
    align_corners_int = 1 if align_corners else 0
    return builder.make_node(
        "GridSample",
        inputs,
        {
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners_int,
        },
        node.name,
    )


def _map_embedding(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Embedding mapping."""
    w = node.inputs.get("W", [])
    ids = node.inputs.get("Ids", [])
    return builder.make_node("Gather", w + ids, {"axis": 0}, node.name)


def _map_affine_grid(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Affine Grid mapping."""
    out_shape = node.inputs.get("OutputShape", [])
    grid = builder.make_node("ConstantOfShape", out_shape, {}, f"{node.name}_grid")[0]
    return [grid]


def _map_im2sequence(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Image to Sequence mapping."""
    x = node.inputs.get("X", [])
    flat = builder.make_node("Flatten", x, {"axis": 1}, f"{node.name}_flat")[0]
    return [flat]


VISION_OPS_MAPPING: dict[
    str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]
] = {
    "bicubic_interp": _map_resize("cubic"),
    "bilinear_interp": _map_resize("linear"),
    "nearest_interp": _map_resize("nearest"),
    "trilinear_interp": _map_resize("linear"),
    "bicubic_interp_v2": _map_resize("cubic"),
    "bilinear_interp_v2": _map_resize("linear"),
    "nearest_interp_v2": _map_resize("nearest"),
    "trilinear_interp_v2": _map_resize("linear"),
    "generate_proposals": _map_generate_proposals,
    "generate_proposals_v2": _map_generate_proposals,
    "box_coder": _map_box_coder,
    "multiclass_nms": _map_multiclass_nms,
    "multiclass_nms3": _map_multiclass_nms,
    "prior_box": _map_prior_box,
    "density_prior_box": _map_prior_box,
    "distribute_fpn_proposals": _map_generate_proposals,
    "matrix_nms": _map_multiclass_nms,
    "yolo_box": _map_yolo_box,
    "yolo_box_head": _map_yolo_box,
    "grid_sampler": _map_grid_sampler,
    "affine_grid": _map_affine_grid,
    "im2sequence": _map_im2sequence,
    "embedding": _map_embedding,
    "lookup_table": _map_embedding,
    "lookup_table_v2": _map_embedding,
}
