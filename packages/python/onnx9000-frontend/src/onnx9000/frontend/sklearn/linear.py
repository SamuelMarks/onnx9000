from onnx9000.core.ir import Graph, Node, Attribute


def _convert_linear_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("linear_regressor_out")
    node = Node("LinearRegressor", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "coef_"):
        node.attrs["coefficients"] = Attribute(
            "coefficients", "FLOATS", estimator.coef_.flatten().tolist()
        )
    if hasattr(estimator, "intercept_"):
        node.attrs["intercepts"] = Attribute(
            "intercepts", "FLOATS", estimator.intercept_.flatten().tolist()
        )
    graph.nodes.append(node)
    return [out_name]


def convert_linear_regression(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ridge(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ridge_cv(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso_cv(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_elastic_net(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_elastic_net_cv(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lars(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso_lars(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_omp(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_bayesian_ridge(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ard_regression(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_passive_aggressive_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_sgd_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_huber_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_theil_sen_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_quantile_regressor(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_poisson_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_gamma_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_tweedie_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_regressor(estimator, input_names, graph)


def _convert_linear_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_label = graph._uniquify_tensor_name("label")
    out_prob = graph._uniquify_tensor_name("probabilities")
    node = Node(
        "LinearClassifier", domain="ai.onnx.ml", inputs=input_names, outputs=[out_label, out_prob]
    )
    if hasattr(estimator, "coef_"):
        node.attrs["coefficients"] = Attribute(
            "coefficients", "FLOATS", estimator.coef_.flatten().tolist()
        )
    if hasattr(estimator, "intercept_"):
        node.attrs["intercepts"] = Attribute(
            "intercepts", "FLOATS", estimator.intercept_.flatten().tolist()
        )
    if hasattr(estimator, "classes_"):
        node.attrs["multi_class"] = Attribute(
            "multi_class", "INT", 1 if len(estimator.classes_) > 2 else 0
        )
        classes = estimator.classes_
        if len(classes) > 0 and isinstance(classes[0], (str, bytes)):
            node.attrs["classlabels_strings"] = Attribute(
                "classlabels_strings", "STRINGS", [str(c) for c in classes]
            )
        else:
            node.attrs["classlabels_int64s"] = Attribute(
                "classlabels_int64s", "INTS", [int(c) for c in classes]
            )
    zipmap_out = graph._uniquify_tensor_name("zipmap_prob")
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


def convert_logistic_regression(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_logistic_regression_cv(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_passive_aggressive_classifier(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_perceptron(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_ridge_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_ridge_classifier_cv(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_sgd_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_linear_classifier(estimator, input_names, graph)
