"""ONNX9000 JSON Extractor."""

import json
from typing import Any

from onnx9000.core.ir import Graph


def _default_serializer(obj: Any) -> Any:
    """Default JSON serializer that drops raw buffer data."""
    if isinstance(obj, (bytes, bytearray)):
        return f"[Buffer: {len(obj)} bytes]"
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def extract_json(graph: Graph, indent: int = 2) -> str:
    """
    Extract a JSON string representation of an ONNX Graph AST.

    Args:
        graph: The ONNX Graph object
        indent: Formatting spaces for JSON.dumps (default: 2)

    Returns:
        Serialized JSON string
    """
    return json.dumps(graph, default=_default_serializer, indent=indent)
