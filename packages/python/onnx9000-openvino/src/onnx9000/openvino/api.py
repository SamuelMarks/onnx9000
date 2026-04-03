"""API for OpenVINO exporter."""

from onnx9000.core.ir import Graph

from .exporter import OpenVinoExporter


def export_model(
    onnx_model: Graph, precision: str = "fp32", version: str = "11", clamp_dynamic: bool = False
) -> tuple[str, bytes]:
    """Export an ONNX9000 Graph to OpenVINO IR (XML + bin)."""
    compress_to_fp16 = precision.lower() == "fp16"
    exporter = OpenVinoExporter(
        onnx_model, version=version, compress_to_fp16=compress_to_fp16, clamp_dynamic=clamp_dynamic
    )
    return exporter.export()
