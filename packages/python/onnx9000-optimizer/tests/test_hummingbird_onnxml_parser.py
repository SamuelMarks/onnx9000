"""Tests the hummingbird onnxml parser module functionality."""

from unittest.mock import MagicMock, Mock

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.onnxml_parser import (
    apply_onnxml_post_transform,
    extract_tree_ensemble_attributes,
    parse_onnxml_tree_ensemble,
)


def test_parse_onnxml_tree_ensemble() -> None:
    """Tests the parse onnxml tree ensemble functionality."""
    node = Node("TreeEnsembleClassifier")
    # Mocking attributes
    MagicMock()

    # We will just patch the node's attrs directly
    node.attrs["nodes_treeids"] = Mock(value=[0, 0, 0])
    node.attrs["nodes_nodeids"] = Mock(value=[0, 1, 2])
    node.attrs["nodes_featureids"] = Mock(value=[2, 0, 0])
    node.attrs["nodes_values"] = Mock(value=[0.5, 0.0, 0.0])
    node.attrs["nodes_modes"] = Mock(value=[b"BRANCH_LEQ", b"LEAF", b"LEAF"])
    node.attrs["nodes_truenodeids"] = Mock(value=[1, 0, 0])
    node.attrs["nodes_falsenodeids"] = Mock(value=[2, 0, 0])

    trees = parse_onnxml_tree_ensemble(node)

    assert len(trees) == 1
    tree = trees[0]

    assert len(tree.features) == 3
    assert tree.features[0] == 2
    assert tree.thresholds[0] == 0.5
    assert tree.left_children[0] == 1
    assert tree.right_children[0] == 2

    assert tree.left_children[1] == -1
    assert tree.right_children[1] == -1


def test_extract_tree_ensemble_attributes() -> None:
    """Tests the extract tree ensemble attributes functionality."""
    node = Node("TreeEnsembleClassifier")
    node.attrs["nodes_treeids"] = Mock(value=[0])
    node.attrs["class_ids"] = Mock(value=[1, 2])

    attrs = extract_tree_ensemble_attributes(node)
    assert attrs["nodes_treeids"] == [0]
    assert attrs["class_ids"] == [1, 2]


def test_apply_onnxml_post_transform() -> None:
    """Tests the apply onnxml post transform functionality."""
    g = Graph(name="test")
    node = Node("TreeEnsembleClassifier")

    node.attrs["post_transform"] = Mock(value=b"SOFTMAX")
    apply_onnxml_post_transform(g, node)
    assert g.nodes[-1].op_type == "Softmax"

    node.attrs["post_transform"] = Mock(value=b"LOGISTIC")
    apply_onnxml_post_transform(g, node)
    assert g.nodes[-1].op_type == "Sigmoid"
