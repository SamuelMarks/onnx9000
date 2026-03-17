"""Provides naive bayes module functionality."""

from onnx9000.core.ir import Graph, Node


def _convert_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert nb operation."""
    out_label = graph._uniquify_tensor_name("nb_label")
    out_prob = graph._uniquify_tensor_name("nb_prob")
    node = Node(
        "LinearClassifier", domain="ai.onnx.ml", inputs=input_names, outputs=[out_label, out_prob]
    )
    graph.nodes.append(node)
    return [out_label, out_prob]


def convert_gaussian_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert gaussian nb operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_multinomial_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert multinomial nb operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_complement_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert complement nb operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_bernoulli_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert bernoulli nb operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_categorical_nb(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert categorical nb operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_k_neighbors_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Executes the convert k neighbors classifier operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_k_neighbors_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Executes the convert k neighbors regressor operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_radius_neighbors_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Executes the convert radius neighbors classifier operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_radius_neighbors_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Executes the convert radius neighbors regressor operation."""
    return _convert_nb(estimator, input_names, graph)


def convert_nearest_centroid(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Executes the convert nearest centroid operation."""
    return _convert_nb(estimator, input_names, graph)
