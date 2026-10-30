"""Spatial data-layout transformation passes (NCHW <-> NHWC) for ONNX9000 IR graphs."""

from onnx9000.core.ir import Attribute, Graph


def transform_nchw_to_nhwc(graph: Graph) -> bool:
    """Annotate spatial convolution and pooling nodes with `NHWC` layout attributes.

    Args:
        graph: The ONNX9000 Graph to transform in-place.

    Returns:
        `True` if any spatial operators were converted from NCHW to NHWC, `False` otherwise.
    """
    spatial_ops = {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool"}
    modified = False
    for node in graph.nodes:
        if node.op_type in spatial_ops:
            cur = node.attributes.get("data_format")
            cur_val = cur.value if isinstance(cur, Attribute) else cur
            if cur_val != "NHWC":
                node.attributes["data_format"] = Attribute("data_format", "STRING", "NHWC")
                modified = True
    return modified


def transform_nhwc_to_nchw(graph: Graph) -> bool:
    """Annotate spatial convolution and pooling nodes with `NCHW` layout attributes.

    Args:
        graph: The ONNX9000 Graph to transform in-place.

    Returns:
        `True` if any spatial operators were converted from NHWC to NCHW, `False` otherwise.
    """
    spatial_ops = {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool"}
    modified = False
    for node in graph.nodes:
        if node.op_type in spatial_ops:
            cur = node.attributes.get("data_format")
            cur_val = cur.value if isinstance(cur, Attribute) else cur
            if cur_val == "NHWC":
                node.attributes["data_format"] = Attribute("data_format", "STRING", "NCHW")
                modified = True
    return modified
