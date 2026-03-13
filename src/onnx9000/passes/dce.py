"""
Dead Code Elimination (DCE)

Removes nodes whose outputs are never consumed by any other node
and are not in the graph's explicitly defined outputs.
"""

from onnx9000.ir import Graph


def dead_code_elimination(graph: Graph) -> None:
    """Removes dead nodes from the graph."""
    changed = True
    while changed:
        changed = False
        consumed = set(graph.outputs)

        for node in graph.nodes:
            for inp in node.inputs:
                consumed.add(inp)

        new_nodes = []
        for node in graph.nodes:
            if any(out in consumed for out in node.outputs):
                new_nodes.append(node)
            else:
                changed = True

        graph.nodes = new_nodes
