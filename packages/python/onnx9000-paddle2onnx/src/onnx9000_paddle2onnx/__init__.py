"""Backward-compatible re-export module for `onnx9000_paddle2onnx`."""

from onnx9000.paddle2onnx import (
    PaddleGraph,
    PaddleToONNXGraphBuilder,
    convert,
    convert_paddle_to_onnx,
    load_paddle_model,
    paddle_optimize_graph,
)

__all__ = [
    "PaddleGraph",
    "PaddleToONNXGraphBuilder",
    "convert",
    "convert_paddle_to_onnx",
    "load_paddle_model",
    "paddle_optimize_graph",
]
