"""
Quantization Passes

Injects QuantizeLinear/DequantizeLinear (QDQ) nodes for QAT
and performs static INT8 post-training quantization conversions.
"""

from onnx9000.ir import Graph


def insert_qat_nodes(graph: Graph) -> None:
    """Injects FakeQuant nodes for Quantization Aware Training."""
    pass


def convert_to_int8(graph: Graph) -> None:
    """Converts standard FP32 operations to their QLinear equivalents."""
    pass
