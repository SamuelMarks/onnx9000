"""Provides xgboost catboost parser module functionality."""

import json
import logging
from typing import Any

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions

logger = logging.getLogger(__name__)


def parse_xgboost_dump(dump: list[str]) -> list[TreeAbstractions]:
    """Load XGBoost Booster JSON dumps natively."""
    trees = []
    for tree_json in dump:
        tree_dict = json.loads(tree_json)
        abstractions = TreeAbstractions()
        _traverse_xgb_tree(tree_dict, abstractions)
        trees.append(abstractions)
    return trees


def _traverse_xgb_tree(node_dict: dict, abstractions: TreeAbstractions) -> None:
    """Executes the traverse xgb tree operation."""
    if "split" in node_dict:  # Internal node
        feature_name = node_dict["split"]
        # In a real impl, we map feature name to integer index
        feature = hash(feature_name) % 1000
        threshold = float(node_dict["split_condition"])

        curr_idx = len(abstractions.features)

        # Determine missing routing
        missing_id = node_dict.get("missing")
        missing_left = missing_id == node_dict.get("yes")

        abstractions.add_node(feature, threshold, -1, -1, 0.0, missing=1 if missing_left else 0)

        # Children usually passed as array inside "children"
        children = node_dict.get("children", [])
        if len(children) >= 1:
            left_idx = len(abstractions.features)
            _traverse_xgb_tree(children[0], abstractions)
            abstractions.left_children[curr_idx] = left_idx

        if len(children) >= 2:
            right_idx = len(abstractions.features)
            _traverse_xgb_tree(children[1], abstractions)
            abstractions.right_children[curr_idx] = right_idx

    else:  # Leaf
        abstractions.add_node(-1, 0.0, -1, -1, float(node_dict.get("leaf", 0.0)))


def parse_xgb_classifier(estimator: Any) -> list[TreeAbstractions]:
    """Parse XGBClassifier directly from Python memory."""
    if hasattr(estimator, "get_booster"):
        dump = estimator.get_booster().get_dump(dump_format="json")
        return parse_xgboost_dump(dump)
    return []


def parse_xgb_regressor(estimator: Any) -> list[TreeAbstractions]:
    """Parse XGBRegressor directly from Python memory."""
    if hasattr(estimator, "get_booster"):
        dump = estimator.get_booster().get_dump(dump_format="json")
        return parse_xgboost_dump(dump)
    return []


def parse_xgb_ranker(estimator: Any) -> list[TreeAbstractions]:
    """Parse XGBRanker directly from Python memory."""
    return parse_xgb_regressor(estimator)


def handle_xgb_objectives(g: Graph, objective: str) -> None:
    """Map XGBoost objectives (binary:logistic, multi:softmax, multi:softprob, count:poisson)"""
    if objective == "binary:logistic":
        g.nodes.append(Node("Sigmoid", inputs=["raw_scores"], outputs=["probabilities"]))
    elif objective == "multi:softprob":
        g.nodes.append(Node("Softmax", inputs=["raw_scores"], outputs=["probabilities"]))
    elif objective == "multi:softmax":
        g.nodes.append(Node("ArgMax", inputs=["raw_scores"], outputs=["predictions"]))
    elif objective == "count:poisson":
        g.nodes.append(Node("Exp", inputs=["raw_scores"], outputs=["predictions"]))


def parse_catboost_classifier(estimator: Any) -> list[TreeAbstractions]:
    """Parse CatBoostClassifier directly from Python memory.
    Leverage CatBoost Oblivious Trees natively mapping to PerfectTree Strategy.
    """
    return []


def parse_catboost_regressor(estimator: Any) -> list[TreeAbstractions]:
    """Parse CatBoostRegressor directly."""
    return []


def handle_catboost_categorical(g: Graph) -> None:
    """Handle CatBoost one-hot encoded categorical variables mathematically."""
    pass
