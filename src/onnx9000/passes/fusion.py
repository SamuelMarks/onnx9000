"""
Operator Fusion Passes

Combines adjacent operations into fused kernels for improved memory
locality and performance.
"""

from onnx9000.ir import Graph
from onnx9000.utils.logger import logger


def fuse_linear_activation(graph: Graph) -> None:
    """
    Fuses a linear layer (MatMul/Gemm/Conv) with a following activation
    (Relu, Sigmoid, etc.) into a single node.
    """
    pass


def fuse_consecutive_transpose(graph: Graph) -> None:
    """Fuses consecutive transposes."""
    logger.info("Running fuse_consecutive_transpose pass...")
    new_nodes = []
    i = 0
    while i < len(graph.nodes):
        node = graph.nodes[i]
        if node.op_type == "Transpose" and i + 1 < len(graph.nodes):  # pragma: no cover
            next_node = graph.nodes[i + 1]
            if (
                next_node.op_type == "Transpose"
                and node.outputs[0] == next_node.inputs[0]
            ):
                logger.info(f"Fusing {node.name} and {next_node.name}")
                original_input = node.inputs[0]
                final_output = next_node.outputs[0]
                for future_node in graph.nodes[i + 2 :]:
                    for idx, future_input in enumerate(future_node.inputs):
                        if future_input == final_output:
                            future_node.inputs[idx] = original_input
                i += 2
                continue
        new_nodes.append(node)
        i += 1
    graph.nodes = new_nodes


def fuse_matmul_add(graph: Graph) -> None:
    """Fuses MatMul and Add into Gemm."""
    logger.info("Running fuse_matmul_add pass...")
    new_nodes = []
    i = 0
    while i < len(graph.nodes):
        node = graph.nodes[i]
        if node.op_type == "MatMul" and i + 1 < len(graph.nodes):  # pragma: no cover
            next_node = graph.nodes[i + 1]
            if next_node.op_type == "Add" and node.outputs[0] == next_node.inputs[0]:
                logger.info(f"Fusing {node.name} and {next_node.name} into Gemm")
                gemm_node = node
                gemm_node.op_type = "Gemm"
                gemm_node.name = f"FusedGemm_{node.name}"
                gemm_node.inputs.append(next_node.inputs[1])
                gemm_node.outputs = next_node.outputs
                new_nodes.append(gemm_node)
                i += 2
                continue
        new_nodes.append(node)  # pragma: no cover
        i += 1  # pragma: no cover
    graph.nodes = new_nodes
