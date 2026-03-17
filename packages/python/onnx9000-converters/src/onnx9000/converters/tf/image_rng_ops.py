"""Module providing image rng ops functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_resize(mode: str) -> Callable:
    """Executes the  map resize operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node("Resize", node.inputs, {"mode": mode}, node.name)

    return _impl


def _map_crop_and_resize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map crop and resize operation."""
    return builder.make_node("RoiAlign", node.inputs, {}, node.name)


def _map_extract_image_patches(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map extract image patches operation."""
    return builder.make_node("Custom_ExtractImagePatches", node.inputs, {}, node.name)


def _map_random_uniform(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map random uniform operation."""
    return builder.make_node("RandomUniformLike", node.inputs, {}, node.name)


def _map_random_standard_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map random standard normal operation."""
    return builder.make_node("RandomNormalLike", node.inputs, {}, node.name)


def _map_truncated_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map truncated normal operation."""
    return builder.make_node("Custom_TruncatedNormal", node.inputs, {}, node.name)


def _map_multinomial(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map multinomial operation."""
    return builder.make_node("Multinomial", node.inputs, {}, node.name)


def _map_nms(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map nms operation."""
    return builder.make_node("NonMaxSuppression", node.inputs, {}, node.name)


def _map_hsv_to_rgb(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map hsv to rgb operation."""
    return builder.make_node("Custom_HSVToRGB", node.inputs, {}, node.name)


def _map_rgb_to_hsv(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map rgb to hsv operation."""
    return builder.make_node("Custom_RGBToHSV", node.inputs, {}, node.name)


IMAGE_RNG_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "ResizeNearestNeighbor": _map_resize("nearest"),
    "ResizeBilinear": _map_resize("linear"),
    "ResizeBicubic": _map_resize("cubic"),
    "CropAndResize": _map_crop_and_resize,
    "ExtractImagePatches": _map_extract_image_patches,
    "RandomUniform": _map_random_uniform,
    "RandomUniformInt": _map_random_uniform,
    "RandomStandardNormal": _map_random_standard_normal,
    "TruncatedNormal": _map_truncated_normal,
    "Multinomial": _map_multinomial,
    "NonMaxSuppression": _map_nms,
    "NonMaxSuppressionV2": _map_nms,
    "NonMaxSuppressionV3": _map_nms,
    "NonMaxSuppressionV4": _map_nms,
    "NonMaxSuppressionV5": _map_nms,
    "HSVToRGB": _map_hsv_to_rgb,
    "RGBToHSV": _map_rgb_to_hsv,
}
