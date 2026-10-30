"""Broadcasting optimization pass for ONNX9000 IR graphs.

Simplifies redundant Expand/Tile broadcasting nodes feeding elementwise binary operators
that natively support NumPy/ONNX multidirectional broadcasting.
"""

from onnx9000.core.ir import Graph, Tensor


def optimize_broadcasting(graph: Graph) -> bool:
    """Eliminate redundant `Expand` nodes that feed directly into broadcast-capable binary ops.

    Args:
        graph: The ONNX9000 Graph to optimize in-place.

    Returns:
        `True` if any redundant broadcasting nodes were bypassed, `False` otherwise.
    """
    if not graph.nodes:
        return False

    broadcast_ops = {"Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "Less"}
    expand_map: dict[str, str] = {}
    for node in graph.nodes:
        if node.op_type == "Expand" and len(node.inputs) >= 1 and len(node.outputs) == 1:
            in_0 = node.inputs[0]
            out_0 = node.outputs[0]
            in_name = in_0.name if isinstance(in_0, Tensor) else in_0
            out_name = out_0.name if isinstance(out_0, Tensor) else out_0
            expand_map[out_name] = in_name

    if not expand_map:
        return False

    modified = False
    for node in graph.nodes:
        if node.op_type in broadcast_ops:
            new_inputs = []
            for inp in node.inputs:
                inp_name = inp.name if isinstance(inp, Tensor) else inp
                if inp_name in expand_map:
                    new_inputs.append(expand_map[inp_name])
                    modified = True
                else:
                    new_inputs.append(inp)
            node.inputs = new_inputs

    return modified
