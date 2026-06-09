"""Graph Fusion."""

from ml_switcheroo_ir import LogicalGraph, topological_sort


def fuse_elementwise(graph: LogicalGraph) -> LogicalGraph:
    """Implement kernel fusion combining sequential elementwise operations."""
    # Simplified mock implementation
    new_graph = LogicalGraph(
        name=f"{graph.name}_fused", outputs=list(graph.outputs), mesh=graph.mesh
    )

    # Just copy the nodes for the mock
    for node in topological_sort(graph):
        new_graph.nodes[node.id] = node
        if node.op_type in ["Add", "Mul"]:
            node.attributes["fused"] = True

    return new_graph
