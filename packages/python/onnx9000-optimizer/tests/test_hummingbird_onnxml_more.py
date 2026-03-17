"""Tests the hummingbird onnxml more module functionality."""

import pytest
from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.onnxml_parser import (
    apply_onnxml_post_transform,
    ensure_static_shapes,
    parse_onnxml_binarizer,
    parse_onnxml_category_mapper,
    parse_onnxml_feature_extractor,
    parse_onnxml_imputer,
    parse_onnxml_linear,
    parse_onnxml_normalizer,
    parse_onnxml_onehot,
    parse_onnxml_scaler,
    parse_onnxml_svm,
    parse_onnxml_zipmap,
)


def test_hummingbird_onnxml_stubs():
    """Tests the hummingbird onnxml stubs functionality."""
    n = Node("dummy", [], [])
    g = Graph("g")
    parse_onnxml_linear(n)
    parse_onnxml_svm(n)
    parse_onnxml_scaler(n)
    parse_onnxml_normalizer(n)
    parse_onnxml_binarizer(n)
    parse_onnxml_onehot(n)
    parse_onnxml_imputer(n)
    parse_onnxml_feature_extractor(n)
    parse_onnxml_category_mapper(n)
    parse_onnxml_zipmap(n)
    apply_onnxml_post_transform(g, n)
    ensure_static_shapes(g)


from onnx9000.core.ir import Attribute
from onnx9000.optimizer.hummingbird.onnxml_parser import (
    extract_tree_ensemble_attributes,
    parse_onnxml_tree_ensemble,
)


def test_parse_onnxml_tree_ensemble():
    """Tests the parse onnxml tree ensemble functionality."""
    n = Node("TreeEnsembleClassifier", [], [])
    res = parse_onnxml_tree_ensemble(n)
    assert len(res) == 0

    n.attributes["nodes_treeids"] = Attribute("nodes_treeids", value=[0, 0, 1])
    n.attributes["nodes_nodeids"] = Attribute("nodes_nodeids", value=[0, 1, 0])
    n.attributes["nodes_featureids"] = Attribute("nodes_featureids", value=[0, 1, 2])
    n.attributes["nodes_values"] = Attribute("nodes_values", value=[0.5, 0.0, 1.0])
    n.attributes["nodes_modes"] = Attribute("nodes_modes", value=[b"BRANCH_LEQ", b"LEAF", b"LEAF"])
    n.attributes["nodes_truenodeids"] = Attribute("nodes_truenodeids", value=[1, -1, -1])
    n.attributes["nodes_falsenodeids"] = Attribute("nodes_falsenodeids", value=[2, -1, -1])

    pass  # because it expects node.attributes
    res = parse_onnxml_tree_ensemble(n)
    assert len(res) == 2


def test_extract_tree_ensemble_attributes():
    """Tests the extract tree ensemble attributes functionality."""
    n = Node("TreeEnsembleClassifier", [], [])
    n.attributes["class_ids"] = Attribute("class_ids", value=[1])
    pass
    res = extract_tree_ensemble_attributes(n)
    assert res["class_ids"] == [1]


def test_apply_onnxml_post_transform():
    """Tests the apply onnxml post transform functionality."""
    g = Graph("g")
    n = Node("dummy", [], [])
    n.attributes["post_transform"] = Attribute("post_transform", value=b"SOFTMAX")
    apply_onnxml_post_transform(g, n)
    assert g.nodes[-1].op_type == "Softmax"

    n2 = Node("dummy", [], [])
    n2.attributes["post_transform"] = Attribute("post_transform", value=b"LOGISTIC")
    apply_onnxml_post_transform(g, n2)
    assert g.nodes[-1].op_type == "Sigmoid"
