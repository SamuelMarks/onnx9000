"""Module providing api functionality."""

import logging

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.control_flow_ops import CONTROL_FLOW_OPS_MAPPING
from onnx9000.converters.tf.image_rng_ops import IMAGE_RNG_OPS_MAPPING
from onnx9000.converters.tf.keras_layers import KERAS_LAYERS_MAPPING
from onnx9000.converters.tf.math_ops import MATH_OPS_MAPPING
from onnx9000.converters.tf.nn_ops import NN_OPS_MAPPING
from onnx9000.converters.tf.parsers import (
    TFGraph,
    load_h5_model,
    load_keras_v3,
    parse_graphdef,
    parse_saved_model,
    parse_tflite,
)
from onnx9000.converters.tf.reduction_ops import REDUCTION_OPS_MAPPING
from onnx9000.converters.tf.tensor_ops import TENSOR_OPS_MAPPING
from onnx9000.converters.tf.tflite_ops import TFLITE_OPS_MAPPING
from onnx9000.core.ir import Graph, Tensor

ALL_MAPPINGS = {}
ALL_MAPPINGS.update(MATH_OPS_MAPPING)
ALL_MAPPINGS.update(NN_OPS_MAPPING)
ALL_MAPPINGS.update(TENSOR_OPS_MAPPING)
ALL_MAPPINGS.update(REDUCTION_OPS_MAPPING)
ALL_MAPPINGS.update(CONTROL_FLOW_OPS_MAPPING)
ALL_MAPPINGS.update(IMAGE_RNG_OPS_MAPPING)
ALL_MAPPINGS.update(KERAS_LAYERS_MAPPING)
ALL_MAPPINGS.update(TFLITE_OPS_MAPPING)


def _convert_tfgraph(tf_graph: TFGraph, name: str = "tf_to_onnx") -> Graph:
    """Execute the  convert tfgraph operation."""
    builder = TFToONNXGraphBuilder(name=name)
    sorted_nodes = tf_graph.topological_sort()
    if not sorted_nodes:
        sorted_nodes = tf_graph.nodes
    tf_graph.resolve_duplicate_names()
    for node in sorted_nodes:
        if node.op == "Placeholder":
            builder.add_constant(node.name, 0, 1, ())
            continue
        mapper = ALL_MAPPINGS.get(node.op)
        if mapper:
            mapper(builder, node)
        else:
            logging.warning(f"Fallback to custom op for unknown node: {node.op}")
            builder.make_node(f"Custom_TF_{node.op}", node.inputs, node.attr, node.name)
    if builder.graph.nodes:
        last_node = builder.graph.nodes[-1]
        for out in last_node.outputs:
            if out in builder.graph.tensors:
                builder.graph.outputs.append(builder.graph.tensors[out])
            else:
                builder.graph.tensors[out] = Tensor(name=out, dtype=1, shape=())
                builder.graph.outputs.append(builder.graph.tensors[out])
    return builder.graph


def convert_tf_to_onnx(model_data: bytes, is_saved_model: bool = False) -> Graph:
    """Convert a TensorFlow GraphDef or SavedModel to ONNX IR."""
    tf_graph = parse_saved_model(model_data) if is_saved_model else parse_graphdef(model_data)
    return _convert_tfgraph(tf_graph, name="tf_graph")


def convert_keras_to_onnx(model_data: bytes, is_v3: bool = False) -> Graph:
    """Convert a Keras H5 or v3 model to ONNX IR."""
    tf_graph = load_keras_v3(model_data) if is_v3 else load_h5_model(model_data)
    return _convert_tfgraph(tf_graph, name="keras_graph")


def convert_tflite_to_onnx(model_data: bytes) -> Graph:
    """Convert a TFLite FlatBuffer to ONNX IR."""
    tf_graph = parse_tflite(model_data)
    return _convert_tfgraph(tf_graph, name="tflite_graph")
