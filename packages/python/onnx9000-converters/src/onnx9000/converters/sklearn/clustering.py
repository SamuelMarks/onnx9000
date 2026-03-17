from onnx9000.core.ir import Graph, Node


def _convert_clustering(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_label = graph._uniquify_tensor_name("cluster_label")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_label])
    graph.nodes.append(node)
    return [out_label]


def convert_kmeans(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_mini_batch_kmeans(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_bisecting_kmeans(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_dbscan(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_optics(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_mean_shift(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_spectral_clustering(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_agglomerative_clustering(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_gaussian_mixture(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)


def convert_bayesian_gaussian_mixture(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_clustering(estimator, input_names, graph)
