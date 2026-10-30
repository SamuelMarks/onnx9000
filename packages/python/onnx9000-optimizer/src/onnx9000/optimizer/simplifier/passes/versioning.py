"""ONNX opset versioning and legacy operator fallback passes for ONNX9000."""

from onnx9000.core.ir import Graph


def apply_opset_fallbacks(graph: Graph) -> bool:
    """Upgrade deprecated legacy ONNX operator names (e.g. `SpatialBN`) to standard equivalents.

    Args:
        graph: The ONNX9000 Graph to update in-place.

    Returns:
        `True` if any legacy operators were rewritten, `False` otherwise.
    """
    legacy_map = {
        "SpatialBN": "BatchNormalization",
        "PadConstant": "Pad",
        "Crop": "Slice",
    }
    modified = False
    for node in graph.nodes:
        if node.op_type in legacy_map:
            node.op_type = legacy_map[node.op_type]
            modified = True
    return modified


def enforce_opset_18(graph: Graph) -> bool:
    """Ensure `graph.opset_imports` targets ONNX opset version 18 for the default domain.

    Args:
        graph: The ONNX9000 Graph to update in-place.

    Returns:
        `True` if the default opset version was updated to 18, `False` if already 18 or empty.
    """
    if not graph.nodes and not graph.opset_imports:
        return False
    current = graph.opset_imports.get("", graph.opset_imports.get("ai.onnx"))
    if current == 18:
        return False
    graph.opset_imports[""] = 18
    return True
