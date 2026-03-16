from onnx9000.core.ir import Graph, Node, Attribute


def _convert_tree_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
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
    if hasattr(estimator, "classes_"):
        classes = estimator.classes_
        if len(classes) > 0 and isinstance(classes[0], (str, bytes)):
            node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", "STRINGS", [str(c) for c in classes]
            )
        else:
            node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", "INTS", [int(c) for c in classes]
            )
    zipmap_out = graph._uniquify_tensor_name("zipmap_tree")
    zipmap_node = Node("ZipMap", domain="ai.onnx.ml", inputs=[out_prob], outputs=[zipmap_out])
    if hasattr(estimator, "classes_"):
        if isinstance(estimator.classes_[0], (str, bytes)):
            zipmap_node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", [str(c) for c in estimator.classes_], "STRINGS"
            )
        else:
            zipmap_node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", [int(c) for c in estimator.classes_], "INTS"
            )
    graph.nodes.extend([node, zipmap_node])
    return [out_label, zipmap_out]


def _convert_tree_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("tree_regressor_out")
    node = Node(
        "TreeEnsembleRegressor", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name]
    )
    node.attrs["nodes_treeids"] = Attribute("nodes_treeids", "INTS", [0])
    node.attrs["nodes_nodeids"] = Attribute("nodes_nodeids", "INTS", [0])
    node.attrs["nodes_featureids"] = Attribute("nodes_featureids", "INTS", [0])
    node.attrs["nodes_values"] = Attribute("nodes_values", "FLOATS", [0.0])
    node.attrs["nodes_modes"] = Attribute("nodes_modes", "STRINGS", ["LEAF"])
    graph.nodes.append(node)
    return [out_name]


def convert_decision_tree_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_decision_tree_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_extra_tree_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_extra_tree_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_random_forest_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_random_forest_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_extra_trees_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_extra_trees_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_gradient_boosting_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_gradient_boosting_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_hist_gradient_boosting_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_hist_gradient_boosting_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_ada_boost_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_ada_boost_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_bagging_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_bagging_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_isolation_forest(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_voting_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_voting_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)


def convert_stacking_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_classifier(estimator, input_names, graph)


def convert_stacking_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_tree_regressor(estimator, input_names, graph)
