"""Tests the hummingbird xgboost more module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import (
    handle_catboost_categorical,
    handle_xgb_objectives,
    parse_catboost_classifier,
    parse_catboost_regressor,
    parse_xgb_classifier,
    parse_xgb_ranker,
    parse_xgb_regressor,
)


class MockBooster:
    """Represents the Mock Booster class."""

    def get_dump(self, dump_format="json"):
        """Execute the get dump operation."""
        return [
            '{"nodeid": 0, "depth": 0, "split": "f0", "split_condition": 1.0, "yes": 1, "no": 2, "missing": 1, "children": [{"nodeid": 1, "leaf": 0.0}, {"nodeid": 2, "leaf": 1.0}]}'
        ]


class MockEstimator:
    """Represents the Mock Estimator class."""

    def __init__(self):
        """Initialize the instance."""
        return None

    def get_booster(self):
        """Execute the get booster operation."""
        return MockBooster()


class MockEstimatorEmpty:
    """Represents the Mock Estimator Empty class."""

    __dummy__ = True


def test_xgboost_stubs():
    """Tests the xgboost stubs functionality."""

    res1 = parse_xgb_classifier(MockEstimator())
    assert len(res1) == 1
    assert len(parse_xgb_classifier(MockEstimatorEmpty())) == 0

    res2 = parse_xgb_regressor(MockEstimator())
    assert len(res2) == 1
    assert len(parse_xgb_regressor(MockEstimatorEmpty())) == 0

    res3 = parse_xgb_ranker(MockEstimator())
    assert len(res3) == 1

    g = Graph("g")
    handle_xgb_objectives(g, "multi:softprob")
    assert g.nodes[-1].op_type == "Softmax"

    handle_xgb_objectives(g, "binary:logistic")
    assert g.nodes[-1].op_type == "Sigmoid"
    handle_xgb_objectives(g, "multi:softmax")
    assert g.nodes[-1].op_type == "ArgMax"
    handle_xgb_objectives(g, "count:poisson")
    assert g.nodes[-1].op_type == "Exp"

    parse_catboost_classifier(None)
    parse_catboost_regressor(None)
    handle_catboost_categorical(g)
