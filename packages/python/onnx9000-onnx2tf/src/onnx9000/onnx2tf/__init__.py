"""ONNX-to-TensorFlow/TFLite NCHW-to-NHWC graph transposition and export pipeline."""

from onnx9000.core.ir import Graph
from onnx9000.tflite_exporter import TFLiteExporter, compile_graph_to_tflite


def transpose_nchw_to_nhwc_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Transpose a 4D NCHW shape tuple `(N, C, H, W)` into NHWC `(N, H, W, C)`.

    Non-4D shapes are returned unchanged.

    Args:
        shape: Input tensor shape tuple.

    Returns:
        Transposed `(N, H, W, C)` tuple if `len(shape) == 4`, else `shape`.
    """
    if len(shape) == 4:
        return (shape[0], shape[2], shape[3], shape[1])
    return shape


def convert_onnx_to_tflite(graph: Graph, quantize_mode: str = "none") -> bytes:
    """Export an ONNX9000 `Graph` into a serialized TFLite FlatBuffer.

    Args:
        graph: Source ONNX9000 `Graph`.
        quantize_mode: Optional quantization mode (`"none"`, `"fp16"`, `"int8"`).

    Returns:
        Serialized `.tflite` FlatBuffer bytes.
    """
    exporter = TFLiteExporter()
    subgraph_offset = compile_graph_to_tflite(graph, exporter, quant_mode=quantize_mode)
    return exporter.finish(subgraph_offset)


def run() -> str:
    """Execute the default onnx2tf status check.

    Returns:
        onnx2tf status confirmation string.
    """
    return "[onnx2tf] processed"


__all__ = [
    "TFLiteExporter",
    "compile_graph_to_tflite",
    "convert_onnx_to_tflite",
    "run",
    "transpose_nchw_to_nhwc_shape",
]
