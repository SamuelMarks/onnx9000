"""Provide clustering module functionality."""

from onnx9000.converters.sklearn.protocols import ClusterEstimatorProtocol
from onnx9000.core.ir import Graph, Node


def _convert_clustering(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert clustering operation."""
    out_label = graph._uniquify_tensor_name("cluster_label")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_label])
    graph.nodes.append(node)
    return [out_label]


def convert_kmeans(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert kmeans operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_mini_batch_kmeans(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert mini batch kmeans operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_bisecting_kmeans(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert bisecting kmeans operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_dbscan(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert dbscan operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_optics(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert optics operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_mean_shift(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert mean shift operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_spectral_clustering(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert spectral clustering operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_agglomerative_clustering(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert agglomerative clustering operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_gaussian_mixture(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert gaussian mixture operation."""
    return _convert_clustering(estimator, input_names, graph)


def convert_bayesian_gaussian_mixture(
    estimator: ClusterEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert bayesian gaussian mixture operation."""
    return _convert_clustering(estimator, input_names, graph)
