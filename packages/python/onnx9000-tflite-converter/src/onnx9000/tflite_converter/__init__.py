"""ONNX to TFLite FlatBuffer converter (`tflite-converter`) for ONNX9000."""

from onnx9000.core.ir import Graph
from onnx9000.tflite_exporter import TFLiteExporter, compile_graph_to_tflite


class TFLiteConverter:
    """High-level converter exporting ONNX9000 `Graph` objects into `.tflite` FlatBuffers."""

    def __init__(self, quant_mode: str = "none") -> None:
        """Initialize the TFLite converter with an optional quantization mode.

        Args:
            quant_mode: Quantization mode (`"none"`, `"fp16"`, `"int8"`).
        """
        self.quant_mode = quant_mode

    def convert(self, graph: Graph) -> bytes:
        """Compile `graph` into a TFLite FlatBuffer byte array.

        Args:
            graph: Source ONNX9000 `Graph`.

        Returns:
            Serialized `.tflite` FlatBuffer bytes.
        """
        exporter = TFLiteExporter()
        subgraph_offset = compile_graph_to_tflite(graph, exporter, quant_mode=self.quant_mode)
        return exporter.finish(subgraph_offset)


def run() -> str:
    """Execute the default tflite-converter status check.

    Returns:
        tflite-converter status confirmation string.
    """
    return "[tflite-converter] processed"


__all__ = ["TFLiteConverter", "TFLiteExporter", "compile_graph_to_tflite", "run"]
