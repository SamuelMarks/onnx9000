import pytest
from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.lightgbm_parser import (
    parse_lgbm_classifier,
    parse_lgbm_regressor,
    parse_lgbm_ranker,
    handle_lgbm_objectives,
    parse_lgbm_categorical,
    apply_lgbm_scaling,
)


class MockBooster:
    def dump_model(self):
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
    def __init__(self):
        self.booster_ = MockBooster()


class MockEstimatorNoBooster:
    pass


def test_parse_lgbm_classifier():
    res = parse_lgbm_classifier(MockEstimator())
    assert len(res) == 1
    res_empty = parse_lgbm_classifier(MockEstimatorNoBooster())
    assert len(res_empty) == 0


def test_parse_lgbm_regressor():
    res = parse_lgbm_regressor(MockEstimator())
    assert len(res) == 1
    res_empty = parse_lgbm_regressor(MockEstimatorNoBooster())
    assert len(res_empty) == 0


def test_parse_lgbm_ranker():
    res = parse_lgbm_ranker(MockEstimator())
    assert len(res) == 1


def test_handle_lgbm_objectives():
    g = Graph("g")
    handle_lgbm_objectives(g, "multiclass")
    assert g.nodes[-1].op_type == "Softmax"

    handle_lgbm_objectives(g, "binary")
    assert g.nodes[-1].op_type == "Sigmoid"

    handle_lgbm_objectives(g, "poisson")
    assert g.nodes[-1].op_type == "Exp"


def test_stubs():
    g = Graph("g")
    parse_lgbm_categorical(g, [1, 2])
    apply_lgbm_scaling(g, 0.5, 0.1)
