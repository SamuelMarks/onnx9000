"""PaddlePaddle converter operations and graph builders."""

from onnx9000.core.ir import Graph


def identity_removal_pass(graph: Graph) -> Graph:
    """identity_removal_pass implementation."""
    new_nodes = []
    replacements: dict[str, str] = {}
    for node in graph.nodes:
        new_inputs = [replacements.get(inp, inp) for inp in node.inputs]
        node.inputs = new_inputs
        if node.op_type == "Identity":
            if node.inputs and node.outputs:
                replacements[node.outputs[0]] = node.inputs[0]
        else:
            new_nodes.append(node)
    graph.nodes = new_nodes
    return graph


def dropout_removal_pass(graph: Graph) -> Graph:
    """dropout_removal_pass implementation."""
    new_nodes = []
    replacements: dict[str, str] = {}
    for node in graph.nodes:
        new_inputs = [replacements.get(inp, inp) for inp in node.inputs]
        node.inputs = new_inputs
        if node.op_type == "Dropout":
            if node.inputs and node.outputs:
                replacements[node.outputs[0]] = node.inputs[0]
        else:
            new_nodes.append(node)
    graph.nodes = new_nodes
    return graph


def dce_pass(graph: Graph) -> Graph:
    """dce_pass implementation."""
    output_names = {t.name for t in graph.outputs}
    consumed: set[str] = set()
    consumed.update(output_names)
    for node in reversed(graph.nodes):
        is_used = False
        for out in node.outputs:
            if out in consumed:
                is_used = True
                break
        if is_used or node.op_type.startswith("Custom_Paddle"):
            consumed.update(node.inputs)
    new_nodes = []
    for node in graph.nodes:
        is_used = False
        for out in node.outputs:
            if out in consumed:
                is_used = True
                break
        if is_used or node.op_type.startswith("Custom_Paddle"):
            new_nodes.append(node)
    graph.nodes = new_nodes
    return graph


def paddle_optimize_graph(graph: Graph) -> Graph:
    """paddle_optimize_graph implementation."""
    graph = identity_removal_pass(graph)
    graph = dropout_removal_pass(graph)
    graph = dce_pass(graph)
    return graph
