"""Provide linear module functionality."""

from onnx9000.converters.sklearn.protocols import LinearEstimatorProtocol
from onnx9000.core.ir import Attribute, Graph, Node


def _convert_linear_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert linear regressor operation."""
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


def convert_linear_regression(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert linear regression operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ridge(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ridge operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ridge_cv(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ridge cv operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert lasso operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso_cv(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert lasso cv operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_elastic_net(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert elastic net operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_elastic_net_cv(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert elastic net cv operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lars(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert lars operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_lasso_lars(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert lasso lars operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_omp(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert omp operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_bayesian_ridge(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert bayesian ridge operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_ard_regression(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ard regression operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_passive_aggressive_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert passive aggressive regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_sgd_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert sgd regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_huber_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert huber regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_theil_sen_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert theil sen regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_quantile_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert quantile regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_poisson_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert poisson regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_gamma_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert gamma regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def convert_tweedie_regressor(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert tweedie regressor operation."""
    return _convert_linear_regressor(estimator, input_names, graph)


def _convert_linear_classifier(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert linear classifier operation."""
    out_label = graph._uniquify_tensor_name("label")
    out_prob = graph._uniquify_tensor_name("probabilities")
    node = Node(
        "LinearClassifier",
        domain="ai.onnx.ml",
        inputs=input_names,
        outputs=[out_label, out_prob],
    )
    if hasattr(estimator, "coef_"):
        node.attrs["coefficients"] = Attribute(
            "coefficients", "FLOATS", estimator.coef_.flatten().tolist()
        )
    if hasattr(estimator, "intercept_"):
        node.attrs["intercepts"] = Attribute(
            "intercepts", "FLOATS", estimator.intercept_.flatten().tolist()
        )
    zipmap_out = graph._uniquify_tensor_name("zipmap_prob")
    zipmap_node = Node("ZipMap", domain="ai.onnx.ml", inputs=[out_prob], outputs=[zipmap_out])
    if hasattr(estimator, "classes_"):
        node.attrs["multi_class"] = Attribute(
            "multi_class", "INT", 1 if len(estimator.classes_) > 2 else 0
        )
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


def convert_logistic_regression(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert logistic regression operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_logistic_regression_cv(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert logistic regression cv operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_passive_aggressive_classifier(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert passive aggressive classifier operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_perceptron(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert perceptron operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_ridge_classifier(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ridge classifier operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_ridge_classifier_cv(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert ridge classifier cv operation."""
    return _convert_linear_classifier(estimator, input_names, graph)


def convert_sgd_classifier(
    estimator: LinearEstimatorProtocol, input_names: list[str], graph: Graph
) -> list[str]:
    """Execute the convert sgd classifier operation."""
    return _convert_linear_classifier(estimator, input_names, graph)
