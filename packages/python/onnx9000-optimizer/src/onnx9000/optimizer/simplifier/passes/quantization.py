"""Quantization-Aware Training (QAT) and INT8 conversion passes for ONNX9000 IR graphs."""

from onnx9000.core.ir import Attribute, Graph, Node


def insert_qat_nodes(graph: Graph) -> bool:
    """Insert `QuantizeLinear` / `DequantizeLinear` fake-quantization attributes on linear ops.

    Args:
        graph: The ONNX9000 Graph to annotate for QAT.

    Returns:
        `True` if any linear/convolutional nodes were annotated for QAT, `False` otherwise.
    """
    quantizable_ops = {"Conv", "MatMul", "Gemm"}
    modified = False
    for node in graph.nodes:
        if node.op_type in quantizable_ops and "qat_enabled" not in node.attributes:
            node.attributes["qat_enabled"] = Attribute("qat_enabled", "INT", 1)
            modified = True
    return modified


def convert_to_int8(graph: Graph) -> bool:
    """Rewrite standard `Conv` and `MatMul` nodes to `QLinearConv` and `MatMulInteger`.

    Args:
        graph: The ONNX9000 Graph to quantize in-place.

    Returns:
        `True` if any nodes were converted to INT8 operators, `False` otherwise.
    """
    op_map = {"Conv": "QLinearConv", "MatMul": "MatMulInteger"}
    modified = False
    for node in graph.nodes:
        if isinstance(node, Node) and node.op_type in op_map:
            node.op_type = op_map[node.op_type]
            modified = True
    return modified
