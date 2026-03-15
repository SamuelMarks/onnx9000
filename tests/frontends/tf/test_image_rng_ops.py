import pytest
from onnx9000.frontends.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontends.tf.parsers import TFNode
from onnx9000.frontends.tf.image_rng_ops import IMAGE_RNG_OPS_MAPPING


def test_image_ops_resize():
    builder = TFToONNXGraphBuilder()
    outs = IMAGE_RNG_OPS_MAPPING["ResizeNearestNeighbor"](
        builder, TFNode("n1", "ResizeNearestNeighbor", inputs=["a", "b"])
    )
    assert builder.graph.nodes[-1].op_type == "Resize"
    assert builder.graph.nodes[-1].attributes["mode"] == "nearest"
    outs = IMAGE_RNG_OPS_MAPPING["ResizeBilinear"](
        builder, TFNode("n2", "ResizeBilinear", inputs=["a", "b"])
    )
    assert builder.graph.nodes[-1].attributes["mode"] == "linear"
    outs = IMAGE_RNG_OPS_MAPPING["ResizeBicubic"](
        builder, TFNode("n3", "ResizeBicubic", inputs=["a", "b"])
    )
    assert builder.graph.nodes[-1].attributes["mode"] == "cubic"
    outs = IMAGE_RNG_OPS_MAPPING["CropAndResize"](
        builder, TFNode("n4", "CropAndResize", inputs=["a", "b"])
    )
    assert builder.graph.nodes[-1].op_type == "RoiAlign"
    outs = IMAGE_RNG_OPS_MAPPING["ExtractImagePatches"](
        builder, TFNode("n5", "ExtractImagePatches", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Custom_ExtractImagePatches"


def test_rng_ops():
    builder = TFToONNXGraphBuilder()
    outs = IMAGE_RNG_OPS_MAPPING["RandomUniform"](
        builder, TFNode("n", "RandomUniform", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "RandomUniformLike"
    outs = IMAGE_RNG_OPS_MAPPING["RandomUniformInt"](
        builder, TFNode("n", "RandomUniformInt", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "RandomUniformLike"
    outs = IMAGE_RNG_OPS_MAPPING["RandomStandardNormal"](
        builder, TFNode("n", "RandomStandardNormal", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "RandomNormalLike"
    outs = IMAGE_RNG_OPS_MAPPING["TruncatedNormal"](
        builder, TFNode("n", "TruncatedNormal", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Custom_TruncatedNormal"
    outs = IMAGE_RNG_OPS_MAPPING["Multinomial"](
        builder, TFNode("n", "Multinomial", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Multinomial"


def test_image_misc():
    builder = TFToONNXGraphBuilder()
    for op in [
        "NonMaxSuppression",
        "NonMaxSuppressionV2",
        "NonMaxSuppressionV3",
        "NonMaxSuppressionV4",
        "NonMaxSuppressionV5",
    ]:
        outs = IMAGE_RNG_OPS_MAPPING[op](
            builder, TFNode(f"n_{op}", op, inputs=["boxes", "scores"])
        )
        assert builder.graph.nodes[-1].op_type == "NonMaxSuppression"
    outs = IMAGE_RNG_OPS_MAPPING["HSVToRGB"](
        builder, TFNode("n", "HSVToRGB", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Custom_HSVToRGB"
    outs = IMAGE_RNG_OPS_MAPPING["RGBToHSV"](
        builder, TFNode("n", "RGBToHSV", inputs=["a"])
    )
    assert builder.graph.nodes[-1].op_type == "Custom_RGBToHSV"
