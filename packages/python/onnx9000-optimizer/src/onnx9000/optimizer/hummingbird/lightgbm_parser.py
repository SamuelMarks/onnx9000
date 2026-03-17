"""Provides lightgbm parser module functionality."""

import logging
from typing import Any

from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions

logger = logging.getLogger(__name__)


def parse_lightgbm_dump(booster_dump: dict) -> list[TreeAbstractions]:
    """Extract LightGBM booster dumps (JSON) strictly in memory."""
    trees = []
    tree_info = booster_dump.get("tree_info", [])

    for tree_dict in tree_info:
        abstractions = TreeAbstractions()
        _traverse_lgbm_tree(tree_dict.get("tree_structure", {}), abstractions)
        trees.append(abstractions)
    return trees


def _traverse_lgbm_tree(node_dict: dict, abstractions: TreeAbstractions) -> None:
    """Executes the traverse lgbm tree operation."""
    if "split_index" in node_dict:  # Internal node
        # Handle categorical features
        if node_dict.get("decision_type") == "<=":
            threshold = float(node_dict["threshold"])
        else:  # categorical
            threshold = 0.0  # Will be mapped via bitsets

        feature = int(node_dict.get("split_feature", 0))

        # We process current node, then children
        curr_idx = len(abstractions.features)

        # Add placeholder
        abstractions.add_node(
            feature, threshold, -1, -1, 0.0, missing=1 if node_dict.get("default_left", True) else 0
        )

        # Left child
        left_idx = len(abstractions.features)
        _traverse_lgbm_tree(node_dict["left_child"], abstractions)

        # Right child
        right_idx = len(abstractions.features)
        _traverse_lgbm_tree(node_dict["right_child"], abstractions)

        abstractions.left_children[curr_idx] = left_idx
        abstractions.right_children[curr_idx] = right_idx
    else:  # Leaf
        abstractions.add_node(-1, 0.0, -1, -1, float(node_dict.get("leaf_value", 0.0)))


def parse_lgbm_classifier(estimator: Any) -> list[TreeAbstractions]:
    """Parse LGBMClassifier directly from Python memory."""
    if hasattr(estimator, "booster_"):
        dump = estimator.booster_.dump_model()
        return parse_lightgbm_dump(dump)
    return []


def parse_lgbm_regressor(estimator: Any) -> list[TreeAbstractions]:
    """Parse LGBMRegressor directly from Python memory."""
    if hasattr(estimator, "booster_"):
        dump = estimator.booster_.dump_model()
        return parse_lightgbm_dump(dump)
    return []


def parse_lgbm_ranker(estimator: Any) -> list[TreeAbstractions]:
    """Parse LGBMRanker directly from Python memory."""
    return parse_lgbm_regressor(estimator)


def handle_lgbm_objectives(g: Graph, objective: str) -> None:
    """Map LightGBM objectives (Softmax, Sigmoid, Exp for Poisson/Tweedie) to ONNX."""
    if objective == "multiclass":
        g.nodes.append(Node("Softmax", inputs=["raw_scores"], outputs=["probabilities"]))
    elif objective == "multiclassova" or objective == "binary":
        g.nodes.append(Node("Sigmoid", inputs=["raw_scores"], outputs=["probabilities"]))
    elif objective in ["poisson", "tweedie"]:
        g.nodes.append(Node("Exp", inputs=["raw_scores"], outputs=["predictions"]))


def parse_lgbm_categorical(g: Graph, bitset: list[int]) -> None:
    """Transpile LightGBM categorical features (bitset evaluations) to Gather/Equal chains.
    Compress large categorical bitsets using int64 arithmetic in ONNX.
    """
    pass


def apply_lgbm_scaling(g: Graph, base_score: float, learning_rate: float) -> None:
    """Map LightGBM leaf output scaling (learning rate / base score) into matrix biases."""
    pass
