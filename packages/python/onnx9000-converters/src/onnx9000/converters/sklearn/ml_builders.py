"""Provide ml builders module functionality."""

from onnx9000.converters.sklearn.protocols import SKLearnEstimatorProtocol
from onnx9000.core.ir import Graph, Node


def convert_mlp_classifier(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert mlp classifier operation."""
    out_name = graph._uniquify_tensor_name("mlp_classifier_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name, out_name]


def convert_mlp_regressor(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert mlp regressor operation."""
    out_name = graph._uniquify_tensor_name("mlp_regressor_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_k_best(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert select k best operation."""
    out_name = graph._uniquify_tensor_name("select_k_best_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_percentile(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert select percentile operation."""
    out_name = graph._uniquify_tensor_name("select_percentile_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_fpr(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert select fpr operation."""
    out_name = graph._uniquify_tensor_name("select_fpr_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_variance_threshold(
    estimator: SKLearnEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert variance threshold operation."""
    out_name = graph._uniquify_tensor_name("variance_threshold_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]
