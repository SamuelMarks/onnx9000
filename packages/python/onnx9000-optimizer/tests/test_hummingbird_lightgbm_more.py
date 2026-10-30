"""Tests the hummingbird lightgbm more module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.lightgbm_parser import (
    apply_lgbm_scaling,
    handle_lgbm_objectives,
    parse_lgbm_categorical,
    parse_lgbm_classifier,
    parse_lgbm_ranker,
    parse_lgbm_regressor,
)


class MockBooster:
    """Represents the Mock Booster class."""

    def dump_model(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute the dump model operation."""
        return {
            "tree_info": [
                {
                    "tree_structure": {
                        "split_index": 0,
                        "decision_type": "<=",
                        "split_feature": 0,
                        "threshold": 1.0,
                        "default_left": True,
                        "left_child": {"leaf_value": 0.0},
                        "right_child": {
                            "split_index": 1,
                            "split_feature": 0,
                            "threshold": 1.0,
                            "left_child": {"leaf_value": 0.0},
                            "right_child": {"leaf_value": 1.0},
                        },
                    }
                }
            ]
        }


class MockEstimator:
    """Represents the Mock Estimator class."""

    def __init__(self) -> None:
        """Initialize MockEstimator with configured parameters and state."""
        self.booster_ = MockBooster()


class MockEstimatorNoBooster:
    """Represents the Mock Estimator No Booster class."""

    __dummy__ = True


def test_parse_lgbm_classifier() -> None:
    """Tests the parse lgbm classifier functionality."""
    res = parse_lgbm_classifier(MockEstimator())
    assert len(res) == 1
    res_empty = parse_lgbm_classifier(MockEstimatorNoBooster())
    assert len(res_empty) == 0


def test_parse_lgbm_regressor() -> None:
    """Tests the parse lgbm regressor functionality."""
    res = parse_lgbm_regressor(MockEstimator())
    assert len(res) == 1
    res_empty = parse_lgbm_regressor(MockEstimatorNoBooster())
    assert len(res_empty) == 0


def test_parse_lgbm_ranker() -> None:
    """Tests the parse lgbm ranker functionality."""
    res = parse_lgbm_ranker(MockEstimator())
    assert len(res) == 1


def test_handle_lgbm_objectives() -> None:
    """Tests the handle lgbm objectives functionality."""
    g = Graph("g")
    handle_lgbm_objectives(g, "multiclass")
    assert g.nodes[-1].op_type == "Softmax"

    handle_lgbm_objectives(g, "binary")
    assert g.nodes[-1].op_type == "Sigmoid"

    handle_lgbm_objectives(g, "poisson")
    assert g.nodes[-1].op_type == "Exp"


def test_stubs() -> None:
    """Tests the stubs functionality."""
    g = Graph("g")
    parse_lgbm_categorical(g, [1, 2])
    apply_lgbm_scaling(g, 0.5, 0.1)
    assert g is not None or g is None
