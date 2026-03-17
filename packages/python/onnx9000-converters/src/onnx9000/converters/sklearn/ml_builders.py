"""Provides ml builders module functionality."""

from onnx9000.core.ir import Graph, Node


def convert_mlp_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert mlp classifier operation."""
    out_name = graph._uniquify_tensor_name("mlp_classifier_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name, out_name]


def convert_mlp_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert mlp regressor operation."""
    out_name = graph._uniquify_tensor_name("mlp_regressor_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_k_best(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert select k best operation."""
    out_name = graph._uniquify_tensor_name("select_k_best_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_percentile(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert select percentile operation."""
    out_name = graph._uniquify_tensor_name("select_percentile_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_select_fpr(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert select fpr operation."""
    out_name = graph._uniquify_tensor_name("select_fpr_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_variance_threshold(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Executes the convert variance threshold operation."""
    out_name = graph._uniquify_tensor_name("variance_threshold_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]
