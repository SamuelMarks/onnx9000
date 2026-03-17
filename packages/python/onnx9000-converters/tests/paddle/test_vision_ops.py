from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode
from onnx9000.converters.paddle.vision_ops import VISION_OPS_MAPPING


def test_paddle_vision_ops() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "bicubic_interp", inputs={"X": ["a"], "OutSize": ["out"]})
    VISION_OPS_MAPPING["bicubic_interp"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Resize"
    n = PaddleNode("n", "bicubic_interp", inputs={"X": ["a"], "Scale": ["scale"]})
    VISION_OPS_MAPPING["bicubic_interp"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Resize"
    n = PaddleNode("n", "bicubic_interp", inputs={"X": ["a"]}, attrs={"out_h": 10, "out_w": 10})
    VISION_OPS_MAPPING["bicubic_interp"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Resize"
    n = PaddleNode("n", "bicubic_interp", inputs={"X": ["a"]}, attrs={"scale": 2.0})
    VISION_OPS_MAPPING["bicubic_interp"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Resize"
    n = PaddleNode("n", "generate_proposals", inputs={"Scores": ["a"], "BboxDeltas": ["b"]})
    VISION_OPS_MAPPING["generate_proposals"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Shape"
    n = PaddleNode("n", "multiclass_nms", inputs={"BBoxes": ["a"], "Scores": ["b"]})
    VISION_OPS_MAPPING["multiclass_nms"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Shape"
    n = PaddleNode("n", "box_coder", inputs={"PriorBox": ["a"], "TargetBox": ["b"]})
    VISION_OPS_MAPPING["box_coder"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    n = PaddleNode("n", "prior_box", inputs={"Input": ["a"]})
    VISION_OPS_MAPPING["prior_box"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    n = PaddleNode("n", "yolo_box", inputs={"X": ["a"]})
    VISION_OPS_MAPPING["yolo_box"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    n = PaddleNode("n", "grid_sampler", inputs={"X": ["a"], "Grid": ["b"]})
    VISION_OPS_MAPPING["grid_sampler"](builder, n)
    assert builder.graph.nodes[-1].op_type == "GridSample"
    n = PaddleNode(
        "n",
        "grid_sampler",
        inputs={"X": ["a"], "Grid": ["b"]},
        attrs={"align_corners": True, "mode": "nearest"},
    )
    VISION_OPS_MAPPING["grid_sampler"](builder, n)
    assert builder.graph.nodes[-1].op_type == "GridSample"
    n = PaddleNode("n", "embedding", inputs={"W": ["a"], "Ids": ["b"]})
    VISION_OPS_MAPPING["embedding"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Gather"
    n = PaddleNode("n", "affine_grid", inputs={"OutputShape": ["a"]})
    VISION_OPS_MAPPING["affine_grid"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ConstantOfShape"
    n = PaddleNode("n", "im2sequence", inputs={"X": ["a"]})
    VISION_OPS_MAPPING["im2sequence"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Flatten"
