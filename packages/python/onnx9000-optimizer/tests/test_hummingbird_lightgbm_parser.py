import pytest
from unittest.mock import Mock, MagicMock
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.lightgbm_parser import (
    parse_lightgbm_dump,
    parse_lgbm_classifier,
    handle_lgbm_objectives,
)


def test_parse_lightgbm_dump():
    dump = {
        "tree_info": [
            {
                "tree_index": 0,
                "tree_structure": {
                    "split_index": 0,
                    "split_feature": 2,
                    "threshold": 0.5,
                    "decision_type": "<=",
                    "default_left": True,
                    "left_child": {"leaf_value": 1.5},
                    "right_child": {"leaf_value": -1.5},
                },
            }
        ]
    }
    trees = parse_lightgbm_dump(dump)
    assert len(trees) == 1
    tree = trees[0]

    assert len(tree.features) == 3  # root + 2 leaves

    # Check that root was created
    assert tree.features[0] == 2
    assert tree.thresholds[0] == 0.5

    # Because recursion creates placeholder then children, root is at index 0
    # Left child at index 1
    # Right child at index 2

    # Root points to 1 and 2
    assert tree.left_children[0] == 1
    assert tree.right_children[0] == 2

    # Leaves have -1 feature
    assert tree.features[1] == -1
    assert tree.values[1] == 1.5

    assert tree.features[2] == -1
    assert tree.values[2] == -1.5


def test_parse_lgbm_classifier():
    mock_booster = MagicMock()
    mock_booster.dump_model.return_value = {"tree_info": []}

    mock_estimator = Mock()
    mock_estimator.booster_ = mock_booster

    trees = parse_lgbm_classifier(mock_estimator)
    assert len(trees) == 0
    mock_booster.dump_model.assert_called_once()


def test_handle_lgbm_objectives():
    g = Graph(name="test")
    handle_lgbm_objectives(g, "multiclass")
    assert g.nodes[-1].op_type == "Softmax"

    handle_lgbm_objectives(g, "binary")
    assert g.nodes[-1].op_type == "Sigmoid"

    handle_lgbm_objectives(g, "poisson")
    assert g.nodes[-1].op_type == "Exp"
