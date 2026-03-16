from onnx9000.core.ir import Graph, Node, Attribute


def _get_kernel_enum(kernel: str) -> str:
    mapping = {"linear": "LINEAR", "poly": "POLY", "rbf": "RBF", "sigmoid": "SIGMOID"}
    return mapping.get(kernel, "RBF")


def _convert_svm_classifier(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_label = graph._uniquify_tensor_name("svm_label")
    out_prob = graph._uniquify_tensor_name("svm_probabilities")
    node = Node(
        "SVMClassifier", domain="ai.onnx.ml", inputs=input_names, outputs=[out_label, out_prob]
    )
    if hasattr(estimator, "kernel"):
        node.attrs["kernel_type"] = Attribute(
            "kernel_type", "STRING", _get_kernel_enum(estimator.kernel)
        )
    if hasattr(estimator, "support_vectors_"):
        node.attrs["vectors_per_class"] = Attribute(
            "vectors_per_class", "INTS", [len(estimator.support_vectors_)]
        )
        node.attrs["support_vectors"] = Attribute(
            "support_vectors", "FLOATS", estimator.support_vectors_.flatten().tolist()
        )
    if hasattr(estimator, "dual_coef_"):
        node.attrs["coefficients"] = Attribute(
            "coefficients", "FLOATS", estimator.dual_coef_.flatten().tolist()
        )
    if hasattr(estimator, "intercept_"):
        node.attrs["rho"] = Attribute("rho", "FLOATS", estimator.intercept_.flatten().tolist())
    if hasattr(estimator, "probA_") and estimator.probA_.size > 0:
        node.attrs["prob_a"] = Attribute("prob_a", "FLOATS", estimator.probA_.flatten().tolist())
    if hasattr(estimator, "probB_") and estimator.probB_.size > 0:
        node.attrs["prob_b"] = Attribute("prob_b", "FLOATS", estimator.probB_.flatten().tolist())
    if hasattr(estimator, "classes_"):
        node.attrs["classlabels_ints"] = Attribute(
            "classlabels_ints", "INTS", [int(c) for c in estimator.classes_]
        )
    zipmap_out = graph._uniquify_tensor_name("zipmap_svm")
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


def _convert_svm_regressor(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("svm_regressor_out")
    node = Node("SVMRegressor", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "kernel"):
        node.attrs["kernel_type"] = Attribute(
            "kernel_type", "STRING", _get_kernel_enum(estimator.kernel)
        )
    if hasattr(estimator, "support_vectors_"):
        node.attrs["n_supports"] = Attribute("n_supports", "INT", len(estimator.support_vectors_))
        node.attrs["support_vectors"] = Attribute(
            "support_vectors", "FLOATS", estimator.support_vectors_.flatten().tolist()
        )
    if hasattr(estimator, "dual_coef_"):
        node.attrs["coefficients"] = Attribute(
            "coefficients", "FLOATS", estimator.dual_coef_.flatten().tolist()
        )
    if hasattr(estimator, "intercept_"):
        node.attrs["rho"] = Attribute("rho", "FLOATS", estimator.intercept_.flatten().tolist())
    graph.nodes.append(node)
    return [out_name]


def convert_svc(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_classifier(estimator, input_names, graph)


def convert_nusvc(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_classifier(estimator, input_names, graph)


def convert_one_class_svm(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_classifier(estimator, input_names, graph)


def convert_svr(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_regressor(estimator, input_names, graph)


def convert_nusvr(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_regressor(estimator, input_names, graph)


def convert_linear_svc(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_classifier(estimator, input_names, graph)


def convert_linear_svr(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    return _convert_svm_regressor(estimator, input_names, graph)
