"""Headless Graph Modifiers for ONNX models."""

import json
from typing import Union


def rename_input(graph, old_name: str, new_name: str):
    """Rename a graph input and update all node references."""
    for inp in graph.inputs:
        if inp.name == old_name:
            inp.name = new_name

    for node in graph.nodes:
        node.inputs = [new_name if i == old_name else i for i in node.inputs]

    return graph


def change_batch(graph, new_size: int | str):
    """Change the batch size (first dimension) of all inputs and outputs."""
    try:
        new_size_val = int(new_size)
    except ValueError:
        new_size_val = new_size

    for inp in graph.inputs:
        if len(inp.shape) > 0:
            s = list(inp.shape)
            s[0] = new_size_val
            inp.shape = tuple(s)

    for out in graph.outputs:
        if hasattr(out, "shape") and out.shape and len(out.shape) > 0:
            s = list(out.shape)
            s[0] = new_size_val
            out.shape = tuple(s)

    return graph


def mutate(graph, script_path: str):
    """Apply generic mutations from a JSON script."""
    with open(script_path) as f:
        mutations = json.load(f)

    for mut in mutations:
        if mut.get("action") == "remove_node":
            graph.nodes = [n for n in graph.nodes if n.name != mut.get("node_name")]

    return graph
