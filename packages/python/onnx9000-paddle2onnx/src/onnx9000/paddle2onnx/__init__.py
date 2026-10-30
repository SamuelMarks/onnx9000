"""PaddlePaddle to ONNX converter package (`paddle2onnx`) for ONNX9000."""

from onnx9000.converters.paddle.api import convert_paddle_to_onnx, paddle_optimize_graph
from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleGraph, load_paddle_model
from onnx9000.core.exceptions import ConversionError


def convert(model_str: str) -> str:
    """Convert a PaddlePaddle model descriptor string to an ONNX-IR status string.

    Args:
        model_str: Non-empty PaddlePaddle model descriptor.

    Returns:
        Formatted ONNX-IR conversion string.

    Raises:
        ValueError: If `model_str` is empty.
    """
    if not model_str:
        raise ConversionError("Invalid model string")
    return f"[ONNX-IR] from {model_str}"


__all__ = [
    "PaddleGraph",
    "PaddleToONNXGraphBuilder",
    "convert",
    "convert_paddle_to_onnx",
    "load_paddle_model",
    "paddle_optimize_graph",
]
