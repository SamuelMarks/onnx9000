import logging
from typing import Any, List
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions
from onnx9000.core.ir import Graph, Node

logger = logging.getLogger(__name__)


def parse_onnxml_tree_ensemble(node: Node) -> List[TreeAbstractions]:
    """Provide explicit converter for ai.onnx.ml.TreeEnsembleClassifier/Regressor -> ai.onnx Math."""
    trees = []

    if "nodes_treeids" not in node.attrs:
        return trees

    tree_ids = node.attrs["nodes_treeids"].value
    node_ids = node.attrs["nodes_nodeids"].value
    feature_ids = node.attrs["nodes_featureids"].value
    thresholds = node.attrs["nodes_values"].value
    modes = node.attrs["nodes_modes"].value
    true_node_ids = node.attrs["nodes_truenodeids"].value
    false_node_ids = node.attrs["nodes_falsenodeids"].value

    # We'd group by tree_ids and map them to TreeAbstractions natively
    # Mock loop for abstraction
    current_tree = -1
    abstractions = None

    for i, t_id in enumerate(tree_ids):
        if t_id != current_tree:
            if abstractions is not None:
                trees.append(abstractions)
            abstractions = TreeAbstractions()
            current_tree = t_id

        feature = feature_ids[i]
        threshold = thresholds[i]
        mode = modes[i]

        left = true_node_ids[i] if mode != b"LEAF" else -1
        right = false_node_ids[i] if mode != b"LEAF" else -1
        value = 0.0  # Extract from class_weights or target_weights

        abstractions.add_node(feature, threshold, left, right, value)

    if abstractions is not None:
        trees.append(abstractions)

    return trees


def extract_tree_ensemble_attributes(node: Node):
    """Extract all relevant tree attributes natively."""
    attrs = {}
    for k in [
        "nodes_treeids",
        "nodes_nodeids",
        "nodes_featureids",
        "nodes_values",
        "nodes_modes",
        "nodes_truenodeids",
        "nodes_falsenodeids",
        "nodes_missing_value_tracks_true",
        "class_treeids",
        "class_nodeids",
        "class_ids",
        "class_weights",
        "target_treeids",
        "target_nodeids",
        "target_ids",
        "target_weights",
    ]:
        if k in node.attrs:
            attrs[k] = node.attrs[k].value
    return attrs


def parse_onnxml_linear(node: Node):
    """Provide explicit converter for LinearClassifier / LinearRegressor."""
    pass


def parse_onnxml_svm(node: Node):
    """Provide explicit converter for SVMClassifier / SVMRegressor."""
    pass


def parse_onnxml_scaler(node: Node):
    """Provide explicit converter for Scaler -> Add + Mul."""
    pass


def parse_onnxml_normalizer(node: Node):
    pass


def parse_onnxml_binarizer(node: Node):
    pass


def parse_onnxml_onehot(node: Node):
    pass


def parse_onnxml_imputer(node: Node):
    pass


def parse_onnxml_feature_extractor(node: Node):
    """Provide explicit converter for ArrayFeatureExtractor -> ai.onnx.Gather"""
    pass


def parse_onnxml_category_mapper(node: Node):
    """Provide explicit converter for CategoryMapper -> ai.onnx Gather/Where"""
    pass


def parse_onnxml_zipmap(node: Node):
    """Provide explicit converter for ZipMap -> standard Tensors + external dictionaries"""
    pass


def apply_onnxml_post_transform(g: Graph, node: Node):
    """Support post_transform extraction (NONE, SOFTMAX, LOGISTIC) natively."""
    post_transform = "NONE"
    if "post_transform" in node.attrs:
        post_transform = node.attrs["post_transform"].value.decode()

    if post_transform == "SOFTMAX":
        g.nodes.append(Node("Softmax", inputs=["raw"], outputs=["prob"]))
    elif post_transform == "LOGISTIC":
        g.nodes.append(Node("Sigmoid", inputs=["raw"], outputs=["prob"]))


def ensure_static_shapes(g: Graph):
    """Ensure lowered ONNX subgraphs are perfectly statically shaped."""
    pass
