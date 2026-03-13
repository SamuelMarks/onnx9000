"""
Visualization Tools

Exports the internal IR Graph into visualization formats (like Graphviz DOT)
or helps integrate with tools like Netron.
"""

from onnx9000.ir import Graph


def to_dot(graph: Graph) -> str:
    """Converts the graph into a Graphviz DOT string."""
    lines = [f"digraph {graph.name} {{"]
    for node in graph.nodes:
        lines.append(f'  "{node.name}" [label="{node.op_type}"];')
        for inp in node.inputs:
            lines.append(f'  "{inp}" -> "{node.name}";')
        for out in node.outputs:
            lines.append(f'  "{node.name}" -> "{out}";')
    lines.append("}")
    return "\n".join(lines)
