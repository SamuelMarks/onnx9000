"""Module providing passes functionality."""

from onnx9000.core.ir import Graph, Node


def constant_folding_pass(graph: Graph) -> Graph:
    """Execute the constant folding pass operation."""
    try:
        from onnx9000.optimizer.simplifier.passes.constant_folding import constant_folding

        constant_folding(graph)
    except ImportError:
        _ignore = True
    return graph


def identity_removal_pass(graph: Graph) -> Graph:
    """Execute the identity removal pass operation."""
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
    new_outputs = []
    for out in graph.outputs:
        if out.name in replacements:
            continue
        new_outputs.append(out)
    return graph


def dropout_removal_pass(graph: Graph) -> Graph:
    """Execute the dropout removal pass operation."""
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


def remove_debug_nodes_pass(graph: Graph) -> Graph:
    """Execute the remove debug nodes pass operation."""
    debug_ops = {
        "Custom_TFAssert",
        "Custom_TFCheckNumerics",
        "Custom_TFPrint",
        "Custom_TFStopGradient",
    }
    new_nodes = []
    replacements: dict[str, str] = {}
    for node in graph.nodes:
        new_inputs = [replacements.get(inp, inp) for inp in node.inputs]
        node.inputs = new_inputs
        if node.op_type in debug_ops:
            if node.inputs and node.outputs:
                replacements[node.outputs[0]] = node.inputs[0]
        else:
            new_nodes.append(node)
    graph.nodes = new_nodes
    return graph


def transpose_optimizer_pass(graph: Graph) -> Graph:
    """Execute the transpose optimizer pass operation."""
    new_nodes = []
    replacements: dict[str, str] = {}
    node_by_output: dict[str, Node] = {}
    for node in graph.nodes:
        new_inputs = [replacements.get(inp, inp) for inp in node.inputs]
        node.inputs = new_inputs
        if node.op_type == "Transpose" and node.inputs:
            parent_out = node.inputs[0]
            if parent_out in node_by_output:
                parent = node_by_output[parent_out]
                if parent.op_type == "Transpose":
                    perm1 = parent.attributes.get("perm")
                    perm2 = node.attributes.get("perm")
                    if perm1 and perm2:
                        combined_perm = [perm1[p] for p in perm2]
                        if combined_perm == list(range(len(combined_perm))):
                            replacements[node.outputs[0]] = parent.inputs[0]
                            continue
        new_nodes.append(node)
        for out in node.outputs:
            node_by_output[out] = node
    graph.nodes = new_nodes
    return graph


def shape_folding_pass(graph: Graph) -> Graph:
    """Execute the shape folding pass operation."""
    try:
        from onnx9000.optimizer.simplifier.passes.shapes import ShapeInferencePass

        ShapeInferencePass().run(graph)
    except ImportError:
        _ignore = True
    return graph


def pattern_matching_pass(graph: Graph) -> Graph:
    """Execute the pattern matching pass operation."""
    try:
        from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

        run_all_fusions(graph)
    except ImportError:
        _ignore = True
    return graph


def dce_pass(graph: Graph) -> Graph:
    """Execute the dce pass operation."""
    output_names = {t.name for t in graph.outputs}
    consumed: set[str] = set()
    consumed.update(output_names)
    for node in reversed(graph.nodes):
        is_used = False
        for out in node.outputs:
            if out in consumed:
                is_used = True
                break
        if is_used or node.op_type in {"Custom_TFPrint", "Custom_TFAssert"}:
            consumed.update(node.inputs)
    new_nodes = []
    for node in graph.nodes:
        is_used = False
        for out in node.outputs:
            if out in consumed:
                is_used = True
                break
        if is_used or node.op_type in {"Custom_TFPrint", "Custom_TFAssert"}:
            new_nodes.append(node)
    graph.nodes = new_nodes
    return graph


def tf_optimize_graph(graph: Graph) -> Graph:
    """Execute the tf optimize graph operation."""
    graph = identity_removal_pass(graph)
    graph = dropout_removal_pass(graph)
    graph = remove_debug_nodes_pass(graph)
    graph = transpose_optimizer_pass(graph)
    graph = constant_folding_pass(graph)
    graph = shape_folding_pass(graph)
    graph = pattern_matching_pass(graph)
    graph = dce_pass(graph)
    return graph
