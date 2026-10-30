"""Debug probe injection pass for ONNX9000 IR graphs.

Exposes intermediate node outputs as graph outputs so intermediate activations
can be inspected during debugging and differential verification.
"""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor, ValueInfo


def inject_probes(graph: Graph) -> bool:
    """Register intermediate node outputs as debug graph outputs.

    Args:
        graph: The ONNX9000 Graph to instrument in-place.

    Returns:
        `True` if any new debug probe outputs were added to `graph.outputs`, `False` otherwise.
    """
    if not graph.nodes:
        return False

    existing_outputs = {
        out.name if isinstance(out, (Tensor, ValueInfo)) else str(out) for out in graph.outputs
    }
    modified = False
    for node in graph.nodes:
        for out in node.outputs:
            out_name = out.name if isinstance(out, Tensor) else str(out)
            if out_name and out_name not in existing_outputs:
                graph.outputs.append(ValueInfo(name=out_name, shape=(), dtype=DType.FLOAT32))
                existing_outputs.add(out_name)
                modified = True
    return modified
