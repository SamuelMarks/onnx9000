"""ONNX9000 JSON Extractor.

Serializes ONNX9000 IR objects (`Graph`, `Node`, `Tensor`, `ValueInfo`, `Attribute`, `DynamicDim`)
to formatted JSON strings without relying on `__dict__` reflection hacks.
"""

import json
from typing import Protocol, runtime_checkable

from onnx9000.core.ir import (
    Attribute,
    DynamicDim,
    Graph,
    JsonDict,
    JsonValue,
    Node,
    Tensor,
    ValueInfo,
)


@runtime_checkable
class DictSerializableProtocol(Protocol):
    """Protocol for objects that expose an explicit `to_dict()` serialization method."""

    def to_dict(self) -> JsonDict:
        """Convert the instance into a JSON-serializable dictionary."""
        return {}


SerializableType = (
    DictSerializableProtocol
    | Graph
    | Node
    | Tensor
    | ValueInfo
    | Attribute
    | DynamicDim
    | bytes
    | bytearray
    | set[str | int | float]
    | int
    | float
    | str
)


def _default_serializer(obj: SerializableType) -> JsonValue:
    """Serialize non-primitive IR objects, byte buffers, and sets into JSON-compatible values.

    Args:
        obj: Object encountered by `json.dumps`.

    Returns:
        JSON-compatible representation.
    """
    if isinstance(obj, (bytes, bytearray)):
        return f"[Buffer: {len(obj)} bytes]"
    if isinstance(obj, DictSerializableProtocol):
        return obj.to_dict()
    if isinstance(obj, set):
        return [str(x) if not isinstance(x, (int, float, str, bool)) else x for x in obj]
    return str(obj)


def extract_json(graph: SerializableType, indent: int = 2) -> str:
    """Extract a JSON string representation of an ONNX9000 Graph AST or IR object.

    Args:
        graph: The ONNX9000 `Graph` or `DictSerializableProtocol` object.
        indent: Indentation spaces for `json.dumps` (default: 2).

    Returns:
        Serialized JSON string.
    """
    return json.dumps(graph, default=_default_serializer, indent=indent)
