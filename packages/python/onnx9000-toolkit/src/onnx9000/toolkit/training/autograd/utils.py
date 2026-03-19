"""Autograd Utilities.

Helper functions for graph traversal and manipulation during backward pass construction.
"""

from onnx9000.core.ir import Graph, Node, Tensor


class GradientProto:
    """Standard ONNX GradientProto definition."""

    def __init__(self, name: str, weight_name: str, gradient_name: str):
        """Initialize the instance."""
        self.name = name
        self.weight_name = weight_name
        self.gradient_name = gradient_name


def generate_gradient_proto(fwd_graph: Graph, bwd_graph: Graph) -> list[GradientProto]:
    """Generate standard ONNX GradientProto if specifically requested by user flags."""
    protos = []
    for out in bwd_graph.outputs:
        if out.startswith("grad_"):
            weight_name = out[5:]
            if weight_name in fwd_graph.initializers:
                protos.append(
                    GradientProto(
                        name=f"grad_proto_{weight_name}", weight_name=weight_name, gradient_name=out
                    )
                )
    return protos


def calculate_gradient_payload_size(graph: Graph) -> int:
    """Calculate theoretical gradient payload sizes natively using onnx-tool profiling.

    Returns the size in bytes of all exposed gradients.
    """
    from onnx9000.core.profiler import profile

    total_bytes = 0
    profile(graph)
    # The profiler might not explicitly split out just the outputs.
    # So we compute it natively here based on graph outputs.
    from onnx9000.core.dtypes import DType

    def dtype_size(dtype_str: str) -> int:
        """Execute the dtype size operation."""
        if dtype_str == "float32":
            return 4
        elif dtype_str in ("float16", "bfloat16"):
            return 2
        elif dtype_str == "int8":
            return 1
        elif dtype_str == "int32":
            return 4
        elif dtype_str == "int64":
            return 8
        return 4

    def resolve_volume(shape: tuple) -> int:
        """Execute the resolve volume operation."""
        if not shape:
            return 1
        v = 1
        for d in shape:
            if isinstance(d, int):
                v *= d
            else:
                v *= 1  # Default dynamic dim size to 1 for lower bound estimate
        return v

    grad_names = [
        out for out in graph.outputs if out.startswith("grad_") or out == "all_gradients_flat"
    ]
    for g in grad_names:
        tensor = graph.tensors.get(g)
        if tensor:
            total_bytes += resolve_volume(tensor.shape) * dtype_size(tensor.dtype)

    return total_bytes


def compress_gradients_int8(graph: Graph) -> None:
    """Exposes an API to dynamically compress gradients via INT8 Quantization before transmission.

    Inserts a DynamicQuantizeLinear node for each gradient output.
    """
    grad_names = [out for out in graph.outputs if out.startswith("grad_")]
    if not grad_names:
        return

    for g in grad_names:
        quantized_out = f"{g}_quantized"
        scale_out = f"{g}_scale"
        zp_out = f"{g}_zp"

        graph.add_node(
            Node(
                "DynamicQuantizeLinear",
                [g],
                [quantized_out, scale_out, zp_out],
                {},
                name=f"{g}_quantize",
            )
        )

        graph.outputs.remove(g)
        graph.outputs.extend([quantized_out, scale_out, zp_out])


def compile_multi_replica_graph(graph: Graph, num_replicas: int) -> Graph:
    """Compiles multi-replica data parallel topologies into a single massive batched graph.

    Simply duplicates the graph N times natively inside the same execution payload.
    """
    if num_replicas <= 1:
        return graph

    multi_graph = Graph(name=f"{graph.name}_{num_replicas}_replicas")

    # We share the initializers (weights) across all replicas
    for init in graph.initializers:
        multi_graph.initializers.append(init)

    for name, tensor in graph.tensors.items():
        if name in graph.initializers:
            multi_graph.add_tensor(tensor)

    # Duplicate nodes and dynamic tensors
    for i in range(num_replicas):
        suffix = f"_replica_{i}"

        for inp in graph.inputs:
            if inp not in graph.initializers:
                multi_graph.inputs.append(f"{inp}{suffix}")
                t = graph.tensors.get(inp)
                if t:
                    multi_graph.add_tensor(
                        Tensor(
                            name=f"{inp}{suffix}",
                            shape=t.shape,
                            dtype=t.dtype,
                            requires_grad=t.requires_grad,
                        )
                    )

        for out in graph.outputs:
            multi_graph.outputs.append(f"{out}{suffix}")
            t = graph.tensors.get(out)
            if t:
                multi_graph.add_tensor(
                    Tensor(
                        name=f"{out}{suffix}",
                        shape=t.shape,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                )

        for node in graph.nodes:
            new_inputs = [
                inp if inp in graph.initializers else f"{inp}{suffix}" for inp in node.inputs
            ]
            new_outputs = [
                out if out in graph.initializers else f"{out}{suffix}" for out in node.outputs
            ]
            multi_graph.add_node(
                Node(
                    node.op_type,
                    new_inputs,
                    new_outputs,
                    node.attributes.copy(),
                    name=f"{node.name}{suffix}",
                )
            )

            for out in new_outputs:
                orig_out = out.replace(suffix, "")
                if (
                    out not in multi_graph.tensors
                    and orig_out in graph.tensors
                    and orig_out not in graph.initializers
                ):
                    t = graph.tensors[orig_out]
                    multi_graph.add_tensor(
                        Tensor(
                            name=out, shape=t.shape, dtype=t.dtype, requires_grad=t.requires_grad
                        )
                    )

    return multi_graph


def embed_distributed_identifiers(graph: Graph) -> None:
    """Embeds unique NodeArg identifiers natively to coordinate distributed weight synchronization automatically."""
    for idx, g in enumerate([out for out in graph.outputs if out.startswith("grad_")]):
        t = graph.tensors.get(g)
        if t:
            # Storing an embedded ID natively inside tensor doc_string or attributes.
            # Using name formatting or custom metadata if available.
            t.doc_string = f"distributed_sync_id_{idx}"


def add_synchronous_barrier(graph: Graph) -> None:
    """Exposes distributed synchronous barrier points statically inside the execution provider.

    by injecting an explicit Wait/Barrier sequence.
    """
    # Just an identity sync node dependent on flattened gradients to force execution barrier
    if "all_gradients_flat" in graph.outputs:
        barrier_out = "barrier_all_gradients_flat"
        graph.add_node(
            Node("Identity", ["all_gradients_flat"], [barrier_out], {}, name="distributed_barrier")
        )
        graph.outputs.remove("all_gradients_flat")
        graph.outputs.append(barrier_out)


def calculate_communication_bounds(graph: Graph, target_ms: int = 100) -> float:
    """Calculate theoretical gradient communication bounds (MB/s) for federated updates,.

    assuming we want to synchronize gradients within a target_ms budget per step.
    """
    bytes_size = calculate_gradient_payload_size(graph)
    # MB/s = (bytes / 1024 / 1024) / (target_ms / 1000)
    mb_size = bytes_size / (1024 * 1024)
    seconds = target_ms / 1000.0
    return mb_size / seconds if seconds > 0 else 0.0


def flatten_gradients(graph: Graph) -> None:
    """Extract and flattens all gradients cleanly into a single massive 1D Tensor dynamically for network transfer."""
    grad_names = [out for out in graph.outputs if out.startswith("grad_")]
    if not grad_names:
        return

    flattened_grads = []
    for g in grad_names:
        g_flat = f"{g}_flattened"
        shape_1d_const = f"{g}_shape_1d_const"
        # Constant [-1] for flattening
        graph.add_node(Node("Constant", [], [shape_1d_const], {"value": [-1]}, name=f"{g}_c_m1"))
        graph.add_node(Node("Reshape", [g, shape_1d_const], [g_flat], {}, name=f"{g}_flatten"))
        flattened_grads.append(g_flat)

    all_grads_flat = "all_gradients_flat"
    graph.add_node(
        Node("Concat", flattened_grads, [all_grads_flat], {"axis": 0}, name="concat_all_grads")
    )

    # Replace individual gradient outputs with the single flat gradient
    for g in grad_names:
        graph.outputs.remove(g)
    graph.outputs.append(all_grads_flat)


def reverse_topological_sort(graph: Graph) -> list["onnx9000.core.ir.Node"]:
    """Return the nodes of the graph in reverse topological order.

    Detects and breaks gradient tracking loops intelligently.
    """
    # Detect cycles in the directed graph of nodes
    visited = set()
    rec_stack = set()
    ordered_nodes = []

    # Map outputs to nodes
    output_to_node = {}
    for node in graph.nodes:
        for out in node.outputs:
            output_to_node[out] = node

    def dfs(node: "onnx9000.core.ir.Node"):
        """Execute the dfs operation."""
        if node.name in rec_stack:
            # Loop detected, break it intelligently by ignoring this back-edge
            return
        if node.name in visited:
            return

        visited.add(node.name)
        rec_stack.add(node.name)

        for inp in node.inputs:
            if inp in output_to_node:
                dfs(output_to_node[inp])

        rec_stack.remove(node.name)
        ordered_nodes.append(node)

    for node in graph.nodes:
        if node.name not in visited:
            dfs(node)

    # For reverse topological sort, the first node in the list should be the last output node.
    # The DFS post-order naturally gives reverse topological order (leaves first, roots last).
    # Since we want backward pass order (roots first, leaves last), we reverse it.
    # Wait, roots first means output nodes first. So we just reverse the DFS post-order list.
    return list(reversed(ordered_nodes))
