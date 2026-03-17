"""Tests the hummingbird xgboost parser module functionality."""

from unittest.mock import MagicMock, Mock

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import (
    handle_xgb_objectives,
    parse_xgb_classifier,
    parse_xgboost_dump,
)


def test_parse_xgboost_dump() -> None:
    """Tests the parse xgboost dump functionality."""
    dump = [
        """{
            "nodeid": 0,
            "depth": 0,
            "split": "f1",
            "split_condition": 0.5,
            "yes": 1,
            "no": 2,
            "missing": 1,
            "children": [
                {"nodeid": 1, "leaf": 1.5},
                {"nodeid": 2, "leaf": -1.5}
            ]
        }"""
    ]
    trees = parse_xgboost_dump(dump)
    assert len(trees) == 1
    tree = trees[0]

    assert len(tree.features) == 3
    assert tree.thresholds[0] == 0.5
    assert tree.left_children[0] == 1
    assert tree.right_children[0] == 2
    assert tree.features[1] == -1
    assert tree.values[1] == 1.5


def test_parse_xgb_classifier() -> None:
    """Tests the parse xgb classifier functionality."""
    mock_booster = MagicMock()
    mock_booster.get_dump.return_value = []

    mock_estimator = Mock()
    mock_estimator.get_booster.return_value = mock_booster

    trees = parse_xgb_classifier(mock_estimator)
    assert len(trees) == 0
    mock_booster.get_dump.assert_called_once()


def test_handle_xgb_objectives() -> None:
    """Tests the handle xgb objectives functionality."""
    g = Graph(name="test")
    handle_xgb_objectives(g, "binary:logistic")
    assert g.nodes[-1].op_type == "Sigmoid"

    handle_xgb_objectives(g, "multi:softprob")
    assert g.nodes[-1].op_type == "Softmax"

    handle_xgb_objectives(g, "multi:softmax")
    assert g.nodes[-1].op_type == "ArgMax"

    handle_xgb_objectives(g, "count:poisson")
    assert g.nodes[-1].op_type == "Exp"
