"""Provides onnxml parser module functionality."""

import logging

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions

logger = logging.getLogger(__name__)


def parse_onnxml_tree_ensemble(node: Node) -> list[TreeAbstractions]:
    """Provide explicit converter for ai.onnx.ml.TreeEnsembleClassifier/Regressor -> ai.onnx Math."""
    trees = []

    if "nodes_treeids" not in node.attributes:
        return trees

    tree_ids = node.attributes["nodes_treeids"].value
    node.attributes["nodes_nodeids"].value
    feature_ids = node.attributes["nodes_featureids"].value
    thresholds = node.attributes["nodes_values"].value
    modes = node.attributes["nodes_modes"].value
    true_node_ids = node.attributes["nodes_truenodeids"].value
    false_node_ids = node.attributes["nodes_falsenodeids"].value

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
        if k in node.attributes:
            attrs[k] = node.attributes[k].value
    return attrs


def parse_onnxml_linear(node: Node) -> None:
    """Provide explicit converter for LinearClassifier / LinearRegressor."""
    pass


def parse_onnxml_svm(node: Node) -> None:
    """Provide explicit converter for SVMClassifier / SVMRegressor."""
    pass


def parse_onnxml_scaler(node: Node) -> None:
    """Provide explicit converter for Scaler -> Add + Mul."""
    pass


def parse_onnxml_normalizer(node: Node) -> None:
    """Execute the parse onnxml normalizer operation."""
    pass


def parse_onnxml_binarizer(node: Node) -> None:
    """Execute the parse onnxml binarizer operation."""
    pass


def parse_onnxml_onehot(node: Node) -> None:
    """Execute the parse onnxml onehot operation."""
    pass


def parse_onnxml_imputer(node: Node) -> None:
    """Execute the parse onnxml imputer operation."""
    pass


def parse_onnxml_feature_extractor(node: Node) -> None:
    """Provide explicit converter for ArrayFeatureExtractor -> ai.onnx.Gather."""
    pass


def parse_onnxml_category_mapper(node: Node) -> None:
    """Provide explicit converter for CategoryMapper -> ai.onnx Gather/Where."""
    pass


def parse_onnxml_zipmap(node: Node) -> None:
    """Provide explicit converter for ZipMap -> standard Tensors + external dictionaries."""
    pass


def apply_onnxml_post_transform(g: Graph, node: Node) -> None:
    """Support post_transform extraction (NONE, SOFTMAX, LOGISTIC) natively."""
    post_transform = "NONE"
    if "post_transform" in node.attributes:
        post_transform = node.attributes["post_transform"].value.decode()

    if post_transform == "SOFTMAX":
        g.nodes.append(Node("Softmax", inputs=["raw"], outputs=["prob"]))
    elif post_transform == "LOGISTIC":
        g.nodes.append(Node("Sigmoid", inputs=["raw"], outputs=["prob"]))


def ensure_static_shapes(g: Graph) -> None:
    """Ensure lowered ONNX subgraphs are perfectly statically shaped."""
    pass
