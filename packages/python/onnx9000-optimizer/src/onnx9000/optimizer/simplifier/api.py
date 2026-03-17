"""Provides api.py module functionality."""

import logging

from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.constant_folding import constant_folding
from onnx9000.optimizer.simplifier.passes.dce import dead_code_elimination
from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

logger = logging.getLogger(__name__)


def _calculate_graph_size(graph: Graph) -> int:
    size = 0
    for init in graph.initializers:
        t = graph.tensors.get(init)
        if t and getattr(t, "data", None) is not None:
            size += (
                t.data.nbytes
                if hasattr(t.data, "nbytes")
                else len(t.data)
                if hasattr(t.data, "__len__")
                else 0
            )
    return size


def check_disconnected_outputs(graph: Graph):
    producers = {}
    for node in graph.nodes:
        for out in node.outputs:
            name = getattr(out, "name", out)
            producers[name] = node

    input_names = {getattr(inp, "name", inp) for inp in graph.inputs}

    for out in graph.outputs:
        out_name = getattr(out, "name", out)
        visited = set()
        stack = [out_name]
        reaches_input = False
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            if curr in input_names:
                reaches_input = True
                break
            if curr in producers:
                for inp in producers[curr].inputs:
                    stack.append(getattr(inp, "name", inp))

        if not reaches_input:
            logger.warning(
                f"Output '{out_name}' is entirely disconnected from all Graph Inputs (Constant Output)."
            )


def extract_scalars(graph: Graph):
    # Some ops require 1D arrays (like Concat, Slice, Split). But Add, Mul, Pow, etc. broadcast 0D.
    # In ONNX, if an initializer is 1D with size 1, it might be safer to keep it 1D unless we know consumers broadcast it well.
    # Let's conservatively turn 1D size-1 constants into 0D scalars if they only feed into element-wise operations.
    elementwise_ops = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Max",
        "Min",
        "Equal",
        "Greater",
        "Less",
        "Where",
    }
    output_names = {getattr(o, "name", o) for o in graph.outputs}
    for init_name in graph.initializers:
        if init_name in output_names:
            continue
        tensor = graph.tensors.get(init_name)
        if tensor and getattr(tensor, "shape", None) == (1,):
            safe_to_scalar = True
            for node in graph.nodes:
                if init_name in node.inputs:
                    if node.op_type not in elementwise_ops:
                        safe_to_scalar = False
                        break
            if safe_to_scalar:
                tensor.shape = ()
                import numpy as np

                if isinstance(tensor.data, np.ndarray):
                    tensor.data = np.array(tensor.data[0])


def simplify(
    graph: Graph,
    skip_fusions: bool = False,
    skip_constant_folding: bool = False,
    skip_shape_inference: bool = False,
    skip_fuse_bn: bool = False,
    skip_rules: list[str] | None = None,
    dry_run: bool = False,
    max_iterations: int = 10,
    log_json_summary: bool = False,
    size_limit_mb: float = 0.0,
    unused_inputs_to_prune: list[str] | None = None,
    input_shapes: dict[str, list] | None = None,
    strip_metadata: bool = False,
    nodes_to_preserve: list[str] | None = None,
    tensor_types: dict[str, str] | None = None,
    target_opset: int | None = None,
    sort_value_info: bool = False,
) -> Graph:
    """
    Simplifies an ONNX9000 IR Graph using Constant Folding, DCE, and Operator Fusion.

    Args:
        graph: The IR Graph to simplify.
        skip_fusions: If True, skips operator fusion passes.
        skip_constant_folding: If True, skips constant folding.
        skip_shape_inference: If True, skips shape inference.
        skip_fuse_bn: If True, skips batch norm fusion.
        skip_rules: List of specific optimization rules to skip.
        dry_run: If True, operates on a copy of the graph and returns it.
        max_iterations: Maximum number of simplification iterations to prevent infinite loops.
        log_json_summary: If True, logs a JSON summary of changes.
        size_limit_mb: If > 0, tracks model size limits.

    Returns:
        The simplified Graph.
    """
    if skip_rules is None:
        skip_rules = []
    if skip_constant_folding:
        skip_rules.append("constant_folding")
    if skip_fusions:
        skip_rules.append("fusions")
    if skip_fuse_bn:
        skip_rules.append("fuse_bn")

    if dry_run:
        import copy

        graph = copy.deepcopy(graph)

    if input_shapes:
        from onnx9000.core.ir import DynamicDim

        for inp in graph.inputs:
            if inp.name in input_shapes:
                new_shape = []
                for dim in input_shapes[inp.name]:
                    if isinstance(dim, str):
                        new_shape.append(DynamicDim(dim))
                    else:
                        new_shape.append(int(dim))
                inp.shape = tuple(new_shape)
                if inp.name in graph.tensors:
                    graph.tensors[inp.name].shape = tuple(new_shape)
                logger.info(f"Overrode input shape for '{inp.name}' to {inp.shape}")
        for t_name, tensor in graph.tensors.items():
            if t_name in input_shapes:
                new_shape = []
                for dim in input_shapes[t_name]:
                    if isinstance(dim, str):
                        new_shape.append(DynamicDim(dim))
                    else:
                        new_shape.append(int(dim))
                tensor.shape = tuple(new_shape)
                logger.info(f"Overrode tensor shape for '{t_name}' to {tensor.shape}")

    if tensor_types:
        from onnx9000.core.dtypes import DType

        type_mapping = {
            "FLOAT": DType.FLOAT32,
            "FLOAT32": DType.FLOAT32,
            "FLOAT16": DType.FLOAT16,
            "DOUBLE": DType.FLOAT64,
            "FLOAT64": DType.FLOAT64,
            "INT32": DType.INT32,
            "INT64": DType.INT64,
            "INT8": DType.INT8,
            "INT16": DType.INT16,
            "UINT8": DType.UINT8,
            "UINT16": DType.UINT16,
            "UINT32": DType.UINT32,
            "UINT64": DType.UINT64,
            "BOOL": DType.BOOL,
            "BFLOAT16": DType.BFLOAT16,
        }
        for inp in graph.inputs:
            if inp.name in tensor_types:
                t_str = str(tensor_types[inp.name]).upper()
                if t_str in type_mapping:
                    inp.dtype = type_mapping[t_str]
                    if inp.name in graph.tensors:
                        graph.tensors[inp.name].dtype = type_mapping[t_str]
                    logger.info(f"Overrode input type for '{inp.name}' to {inp.dtype}")
        for t_name, tensor in graph.tensors.items():
            if t_name in tensor_types:
                t_str = str(tensor_types[t_name]).upper()
                if t_str in type_mapping:
                    tensor.dtype = type_mapping[t_str]
                    logger.info(f"Overrode tensor type for '{t_name}' to {tensor.dtype}")
        for t_name, tensor in graph.tensors.items():
            if t_name in input_shapes:
                new_shape = []
                for dim in input_shapes[t_name]:
                    if isinstance(dim, str):
                        new_shape.append(DynamicDim(dim))
                    else:
                        new_shape.append(int(dim))
                tensor.shape = tuple(new_shape)
                logger.info(f"Overrode tensor shape for '{t_name}' to {tensor.shape}")

    initial_size = _calculate_graph_size(graph)
    if size_limit_mb > 0 and initial_size > size_limit_mb * 1024 * 1024:
        logger.warning(f"Graph exceeds size limit {size_limit_mb}MB before simplification.")

    logger.info(f"Starting simplification for graph '{graph.name}'")
    initial_nodes = len(graph.nodes)

    node_counts_before = {}
    for node in graph.nodes:
        node_counts_before[node.op_type] = node_counts_before.get(node.op_type, 0) + 1

    iteration = 0
    while True:
        if iteration >= max_iterations:
            logger.warning(
                f"Simplification halted after {max_iterations} iterations to prevent infinite loops."
            )
            break
        logger.info(f"Simplification iteration {iteration}")
        nodes_before = len(graph.nodes)

        if "constant_folding" not in skip_rules:
            constant_folding(graph, size_limit_mb if size_limit_mb > 0 else 10.0)
            extract_scalars(graph)

        dead_code_elimination(graph, unused_inputs_to_prune, nodes_to_preserve)

        if "fusions" not in skip_rules:
            run_all_fusions(graph)

        if "shape_inference" not in skip_rules:
            from onnx9000.core.shape_inference import infer_shapes_and_types

            try:
                infer_shapes_and_types(graph)
            except Exception as e:
                import traceback

                logger.warning(f"Shape inference failed: {e}\n{traceback.format_exc()}")

        nodes_after = len(graph.nodes)
        if nodes_after == nodes_before:
            logger.info("Graph stabilized.")
            break
        iteration += 1

    final_nodes = len(graph.nodes)

    if strip_metadata:
        graph.metadata_props.clear()

    used_domains = {""}
    for node in graph.nodes:
        used_domains.add(node.domain)

    new_opsets = {}
    for domain, version in graph.opset_imports.items():
        if domain in used_domains:
            new_opsets[domain] = version
            if target_opset is not None and domain == "":
                new_opsets[domain] = target_opset
                logger.info(f"Target Opset overriden to {target_opset} for domain ''")

    if target_opset is not None and "" not in new_opsets and "" in used_domains:
        new_opsets[""] = target_opset

    graph.opset_imports = new_opsets

    if sort_value_info:
        graph.inputs.sort(key=lambda x: getattr(x, "name", x))
        graph.outputs.sort(key=lambda x: getattr(x, "name", x))
        graph.value_info.sort(key=lambda x: getattr(x, "name", x))

    if not graph.producer_name.endswith("onnx9000-simplifier"):
        if graph.producer_name:
            graph.producer_name = f"{graph.producer_name}_onnx9000-simplifier"
        else:
            graph.producer_name = "onnx9000-simplifier"

    final_size = _calculate_graph_size(graph)
    check_disconnected_outputs(graph)

    if size_limit_mb > 0 and final_size > size_limit_mb * 1024 * 1024:
        logger.warning(f"Graph exceeds size limit {size_limit_mb}MB after simplification.")

    if log_json_summary:
        import json

        node_counts_after = {}
        for node in graph.nodes:
            node_counts_after[node.op_type] = node_counts_after.get(node.op_type, 0) + 1

        summary = {
            "graph_name": graph.name,
            "nodes_before": initial_nodes,
            "nodes_after": final_nodes,
            "nodes_reduced": initial_nodes - final_nodes,
            "size_bytes_before": initial_size,
            "size_bytes_after": final_size,
            "size_reduced_bytes": initial_size - final_size,
            "iterations": iteration,
            "node_types_before": node_counts_before,
            "node_types_after": node_counts_after,
        }
        logger.info(f"SIMPLIFICATION_SUMMARY: {json.dumps(summary)}")

        md_table = f"""
| Metric | Before | After | Reduction |
|---|---|---|---|
| **Nodes** | {initial_nodes} | {final_nodes} | {initial_nodes - final_nodes} |
| **Size (Bytes)** | {initial_size} | {final_size} | {initial_size - final_size} |
"""
        logger.info(f"\nSimplification Report:{md_table}")

    logger.info(f"Simplification complete. Nodes reduced from {initial_nodes} to {final_nodes}.")
    return graph
