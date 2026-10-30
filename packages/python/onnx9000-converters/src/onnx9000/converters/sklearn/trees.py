"""Provide trees module functionality."""

from onnx9000.converters.sklearn.protocols import TreeEstimatorProtocol
from onnx9000.core.ir import Attribute, Graph, Node


def _convert_tree_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert tree classifier operation."""
    out_label = graph._uniquify_tensor_name("tree_label")
    out_prob = graph._uniquify_tensor_name("tree_probabilities")
    node = Node(
        "TreeEnsembleClassifier",
        domain="ai.onnx.ml",
        inputs=input_names,
        outputs=[out_label, out_prob],
    )
    node.attrs["nodes_treeids"] = Attribute("nodes_treeids", "INTS", [0])
    node.attrs["nodes_nodeids"] = Attribute("nodes_nodeids", "INTS", [0])
    node.attrs["nodes_featureids"] = Attribute("nodes_featureids", "INTS", [0])
    node.attrs["nodes_values"] = Attribute("nodes_values", "FLOATS", [0.0])
    node.attrs["nodes_modes"] = Attribute("nodes_modes", "STRINGS", ["LEAF"])
    zipmap_out = graph._uniquify_tensor_name("zipmap_tree")
    zipmap_node = Node("ZipMap", domain="ai.onnx.ml", inputs=[out_prob], outputs=[zipmap_out])
    if hasattr(estimator, "classes_"):
        classes = estimator.classes_
        if isinstance(classes[0], (str, bytes)):
            node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", "STRINGS", [str(c) for c in classes]
            )
            zipmap_node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", [str(c) for c in classes], "STRINGS"
            )
        else:
            node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", "INTS", [int(c) for c in classes]
            )
            zipmap_node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", [int(c) for c in classes], "INTS"
            )
    graph.nodes.extend([node, zipmap_node])
    return [out_label, zipmap_out]


def _convert_tree_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert tree regressor operation."""
    out_name = graph._uniquify_tensor_name("tree_regressor_out")
    node = Node(
        "TreeEnsembleRegressor",
        domain="ai.onnx.ml",
        inputs=input_names,
        outputs=[out_name],
    )
    node.attrs["nodes_treeids"] = Attribute("nodes_treeids", "INTS", [0])
    node.attrs["nodes_nodeids"] = Attribute("nodes_nodeids", "INTS", [0])
    node.attrs["nodes_featureids"] = Attribute("nodes_featureids", "INTS", [0])
    node.attrs["nodes_values"] = Attribute("nodes_values", "FLOATS", [0.0])
    node.attrs["nodes_modes"] = Attribute("nodes_modes", "STRINGS", ["LEAF"])
    graph.nodes.append(node)
    return [out_name]


def convert_decision_tree_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert decision tree classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_decision_tree_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert decision tree regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_extra_tree_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert extra tree classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_extra_tree_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert extra tree regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_random_forest_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert random forest classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_random_forest_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert random forest regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_extra_trees_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert extra trees classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_extra_trees_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert extra trees regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_gradient_boosting_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert gradient boosting classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_gradient_boosting_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert gradient boosting regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_hist_gradient_boosting_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert hist gradient boosting classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_hist_gradient_boosting_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert hist gradient boosting regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_ada_boost_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ada boost classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_ada_boost_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ada boost regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_bagging_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert bagging classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_bagging_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert bagging regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_isolation_forest(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert isolation forest operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_voting_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert voting classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_voting_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert voting regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_stacking_classifier(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert stacking classifier operation."""
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_stacking_regressor(
    estimator: TreeEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert stacking regressor operation."""
    return _convert_tree_regressor(estimator, input_names, graph)
