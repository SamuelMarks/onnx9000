"""Provide decomposition module functionality."""

from onnx9000.core.ir import Graph, Node


def _convert_pca_like(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert pca like operation."""
    matmul_out = graph._uniquify_tensor_name("pca_matmul_out")
    add_out = graph._uniquify_tensor_name("pca_add_out")
    node_matmul = Node("MatMul", domain="", inputs=input_names, outputs=[matmul_out])
    node_add = Node("Add", domain="", inputs=[matmul_out], outputs=[add_out])
    graph.nodes.extend([node_matmul, node_add])
    return [add_out]


def convert_pca(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert pca operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_incremental_pca(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert incremental pca operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_truncated_svd(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert truncated svd operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_fast_ica(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert fast ica operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_nmf(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert nmf operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_kernel_pca(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    """Execute the convert kernel pca operation."""
    return _convert_pca_like(estimator, input_names, graph)


def convert_latent_dirichlet_allocation(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert latent dirichlet allocation operation."""
    return _convert_pca_like(estimator, input_names, graph)
