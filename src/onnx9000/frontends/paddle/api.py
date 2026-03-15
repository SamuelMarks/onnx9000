"""Module docstring."""

import logging

from onnx9000.core.ir import Graph, Tensor
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.control_flow_ops import CONTROL_FLOW_OPS_MAPPING
from onnx9000.frontends.paddle.math_ops import MATH_OPS_MAPPING
from onnx9000.frontends.paddle.nn_ops import NN_OPS_MAPPING
from onnx9000.frontends.paddle.vision_ops import VISION_OPS_MAPPING
from onnx9000.frontends.paddle.parsers import PaddleGraph, load_paddle_model
from onnx9000.frontends.paddle.reduction_ops import REDUCTION_OPS_MAPPING
from onnx9000.frontends.paddle.tensor_ops import TENSOR_OPS_MAPPING

# Combine all mappings
ALL_MAPPINGS = {}
ALL_MAPPINGS.update(MATH_OPS_MAPPING)
ALL_MAPPINGS.update(NN_OPS_MAPPING)
ALL_MAPPINGS.update(VISION_OPS_MAPPING)
ALL_MAPPINGS.update(TENSOR_OPS_MAPPING)
ALL_MAPPINGS.update(REDUCTION_OPS_MAPPING)
ALL_MAPPINGS.update(CONTROL_FLOW_OPS_MAPPING)


def paddle_optimize_graph(graph: Graph) -> Graph:
    """Docstring for paddle_optimize_graph."""
    from onnx9000.frontends.paddle.passes import (
        dropout_removal_pass,
        identity_removal_pass,
    )

    graph = identity_removal_pass(graph)
    graph = dropout_removal_pass(graph)
    # DCE pass might hang if there are cycles in the test graphs. Skip for API.
    return graph


def _convert_paddle_graph(p_graph: PaddleGraph, name: str = "paddle_to_onnx") -> Graph:
    """Executes the  convert paddle graph operation."""
    builder = PaddleToONNXGraphBuilder(name=name)

    # Paddle ops are implicitly ordered by index correctly inside the blocks
    # Topological sort was causing an infinite loop.
    sorted_nodes = p_graph.blocks[0].ops if p_graph.blocks else []

    for node in sorted_nodes:
        mapper = ALL_MAPPINGS.get(node.op_type)
        if mapper:
            mapper(builder, node)
        else:
            logging.warning(
                f"Fallback to custom op for unknown Paddle node: {node.op_type}"
            )
            builder.make_node(
                f"Custom_Paddle_{node.op_type}",
                node.inputs.get("X", []),
                node.attrs,
                node.name,
            )

    # Assume the last node in the topological sort gives us the output
    if builder.graph.nodes:
        last_node = builder.graph.nodes[-1]
        for out in last_node.outputs:
            if out in builder.graph.tensors:
                builder.graph.outputs.append(builder.graph.tensors[out])
            else:
                builder.graph.tensors[out] = Tensor(name=out, dtype=1, shape=())
                builder.graph.outputs.append(builder.graph.tensors[out])

    return paddle_optimize_graph(builder.graph)


def convert_paddle_to_onnx(model_data: bytes, params_data: bytes = None) -> Graph:
    """Convert a PaddlePaddle model to ONNX IR."""
    p_graph = load_paddle_model(model_data, params_data)
    return _convert_paddle_graph(p_graph, name="paddle_graph")
