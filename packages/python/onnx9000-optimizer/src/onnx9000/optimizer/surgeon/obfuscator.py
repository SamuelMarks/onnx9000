"""Model obfuscation for onnx9000."""

from onnx9000.core.ir import Graph


def obfuscate_names(graph: Graph) -> Graph:
    """Rename tensors and nodes to protect intellectual property."""
    tensor_map = {}
    node_counter = 0
    tensor_counter = 0

    # Obfuscate internal tensors
    for name, tensor in list(graph.tensors.items()):
        # Don't obfuscate inputs and outputs to keep the interface functional
        if any(name == i.name for i in graph.inputs) or any(name == o.name for o in graph.outputs):
            continue

        new_name = f"t_{tensor_counter}"
        tensor_counter += 1
        tensor_map[name] = new_name
        tensor.name = new_name

    # Update graph.tensors dict
    new_tensors = {}
    for name, tensor in graph.tensors.items():
        new_tensors[tensor.name] = tensor
    graph.tensors = new_tensors

    # Obfuscate nodes and their input/output references
    for node in graph.nodes:
        node.name = f"n_{node_counter}"
        node_counter += 1

        node.inputs = [tensor_map.get(i, i) for i in node.inputs]
        # outputs are usually Tensor objects, their names are already updated

    return graph
