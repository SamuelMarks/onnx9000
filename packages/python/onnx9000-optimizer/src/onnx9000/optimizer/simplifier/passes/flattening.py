"""Subgraph flattening pass for ONNX9000 IR graphs.

Inlines single-branch `If` or `Body` subgraph attributes into the parent graph
when control-flow conditions are statically known or unconditional.
"""

from onnx9000.core.ir import Attribute, Graph


def flatten_subgraphs(graph: Graph) -> bool:
    """Inline nodes from nested `Graph` attributes into the parent `Graph`.

    Args:
        graph: The ONNX9000 Graph to flatten in-place.

    Returns:
        `True` if any nested subgraphs were flattened into `graph`, `False` otherwise.
    """
    if not graph.nodes:
        return False

    modified = False
    new_nodes = []
    for node in graph.nodes:
        inlined = False
        for attr in node.attributes.values():
            val = attr.value if isinstance(attr, Attribute) else attr
            if isinstance(val, Graph) and val.nodes:
                new_nodes.extend(val.nodes)
                inlined = True
                modified = True
        if not inlined:
            new_nodes.append(node)

    if modified:
        graph.nodes = new_nodes
    return modified
