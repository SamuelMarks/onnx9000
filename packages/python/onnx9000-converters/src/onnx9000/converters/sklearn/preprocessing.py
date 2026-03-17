from onnx9000.core.ir import Attribute, Graph, Node


def convert_standard_scaler(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("standard_scaler_out")
    node = Node("Scaler", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "mean_"):
        node.attrs["offset"] = Attribute("offset", "FLOATS", estimator.mean_.tolist())
    if hasattr(estimator, "scale_"):
        node.attrs["scale"] = Attribute("scale", "FLOATS", estimator.scale_.tolist())
    graph.nodes.append(node)
    return [out_name]


def convert_min_max_scaler(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("minmax_scaler_out")
    node = Node("Scaler", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "min_") and hasattr(estimator, "scale_"):
        node.attrs["offset"] = Attribute(
            "offset", "FLOATS", (-estimator.min_ / estimator.scale_).tolist()
        )
        node.attrs["scale"] = Attribute("scale", "FLOATS", estimator.scale_.tolist())
    graph.nodes.append(node)
    return [out_name]


def convert_max_abs_scaler(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("maxabs_scaler_out")
    node = Node("Scaler", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "scale_"):
        node.attrs["scale"] = Attribute("scale", "FLOATS", (1.0 / estimator.scale_).tolist())
    graph.nodes.append(node)
    return [out_name]


def convert_robust_scaler(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("robust_scaler_out")
    node = Node("Scaler", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    if hasattr(estimator, "center_"):
        node.attrs["offset"] = Attribute("offset", "FLOATS", estimator.center_.tolist())
    if hasattr(estimator, "scale_"):
        node.attrs["scale"] = Attribute("scale", "FLOATS", estimator.scale_.tolist())
    graph.nodes.append(node)
    return [out_name]


def convert_normalizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("normalizer_out")
    node = Node("Normalizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    norm = estimator.norm
    if norm == "l1":
        node.attrs["norm"] = Attribute("norm", "STRING", "L1")
    elif norm == "l2":
        node.attrs["norm"] = Attribute("norm", "STRING", "L2")
    elif norm == "max":
        node.attrs["norm"] = Attribute("norm", "STRING", "MAX")
    graph.nodes.append(node)
    return [out_name]


def convert_binarizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("binarizer_out")
    node = Node("Binarizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    node.attrs["threshold"] = Attribute("threshold", "FLOAT", float(estimator.threshold))
    graph.nodes.append(node)
    return [out_name]


def convert_one_hot_encoder(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("ohe_out")
    node = Node("OneHotEncoder", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    cats_int = []
    cats_str = []
    for cat in estimator.categories_:
        if len(cat) > 0 and isinstance(cat[0], (str, bytes)):
            cats_str.extend([str(x) for x in cat])
        else:
            cats_int.extend([int(x) for x in cat])
    if cats_int:
        node.attrs["cats_int64s"] = Attribute("cats_int64s", "INTS", cats_int)
    if cats_str:
        node.attrs["cats_strings"] = Attribute("cats_strings", "STRINGS", cats_str)
    graph.nodes.append(node)
    return [out_name]


def convert_label_encoder(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("label_encoder_out")
    node = Node("LabelEncoder", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    classes = estimator.classes_
    if len(classes) > 0 and isinstance(classes[0], (str, bytes)):
        node.attrs["classes_strings"] = Attribute(
            "classes_strings", "STRINGS", [str(c) for c in classes]
        )
        node.attrs["default_string"] = Attribute("default_string", "STRING", "")
    else:
        node.attrs["classes_int64s"] = Attribute(
            "classes_int64s", "INTS", [int(c) for c in classes]
        )
        node.attrs["default_int64"] = Attribute("default_int64", "INT", -1)
    graph.nodes.append(node)
    return [out_name]


def convert_ordinal_encoder(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("ordinal_encoder_out")
    node = Node("CategoryMapper", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_polynomial_features(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("poly_features_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_power_transformer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("power_transform_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_quantile_transformer(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("quantile_transform_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_kbins_discretizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("kbins_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_label_binarizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("label_binarizer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_multi_label_binarizer(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("multi_label_binarizer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_simple_imputer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("simple_imputer_out")
    node = Node("Imputer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_missing_indicator(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("missing_indicator_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_iterative_imputer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("iterative_imputer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_knn_imputer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("knn_imputer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_function_transformer(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("function_transformer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_spline_transformer(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("spline_transformer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_dict_vectorizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("dict_vectorizer_out")
    node = Node("DictVectorizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_feature_hasher(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("feature_hasher_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_count_vectorizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("count_vectorizer_out")
    node = Node("CountVectorizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_tfidf_transformer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("tfidf_transformer_out")
    node = Node("TfIdfVectorizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_tfidf_vectorizer(estimator: object, input_names: list[str], graph: Graph) -> list[str]:
    out_name = graph._uniquify_tensor_name("tfidf_vectorizer_out")
    node = Node("TfIdfVectorizer", domain="ai.onnx.ml", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]


def convert_hashing_vectorizer(
    estimator: object, input_names: list[str], graph: Graph
) -> list[str]:
    out_name = graph._uniquify_tensor_name("hashing_vectorizer_out")
    node = Node("Identity", domain="", inputs=input_names, outputs=[out_name])
    graph.nodes.append(node)
    return [out_name]
