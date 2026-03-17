"""CatBoost JSON parser for pure-Python ONNX conversion."""

import json
from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_catboost_json(json_str: str) -> Graph:
    """Parse a CatBoost JSON dump and return an ONNX Graph."""
    data = json.loads(json_str)
    return parse_catboost_dict(data)


def parse_catboost_dict(data: dict[str, Any]) -> Graph:
    """Parse a CatBoost dict representation."""
    graph = Graph("CatBoost_Model")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    num_features = data.get("features_info", {}).get("float_features", [])
    input_vi = ValueInfo("X", DType.FLOAT32, ["batch_size", len(num_features) or 1])
    graph.inputs.append(input_vi)

    objective = data.get("model_info", {}).get("loss_function", "RMSE")

    nodes_treeids = []
    nodes_nodeids = []
    nodes_featureids = []
    nodes_values = []
    nodes_hitrates = []
    nodes_modes = []
    nodes_truenodeids = []
    nodes_falsenodeids = []
    nodes_missing_value_tracks_true = []

    target_treeids = []
    target_nodeids = []
    target_ids = []
    target_weights = []

    trees = data.get("oblivious_trees", [])
    for tree_id, tree in enumerate(trees):
        splits = tree.get("splits", [])
        leaf_values = tree.get("leaf_values", [])
        tree.get("leaf_weights", [])

        depth = len(splits)
        num_leaves = 2**depth

        for leaf_idx in range(num_leaves):
            node_id = (2**depth - 1) + leaf_idx
            target_treeids.append(tree_id)
            target_nodeids.append(node_id)
            target_ids.append(0)
            target_weights.append(leaf_values[leaf_idx] if leaf_idx < len(leaf_values) else 0.0)

            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)
            nodes_featureids.append(0)
            nodes_values.append(0.0)
            nodes_hitrates.append(1.0)
            nodes_modes.append(b"LEAF")
            nodes_truenodeids.append(0)
            nodes_falsenodeids.append(0)
            nodes_missing_value_tracks_true.append(0)

        for d in range(depth - 1, -1, -1):
            split = splits[d]
            feature_idx = split.get("float_feature_index", 0)
            threshold = split.get("border", 0.0)
            num_nodes = 2**d
            for n in range(num_nodes):
                node_id = (2**d - 1) + n
                left_id = 2 * node_id + 1
                right_id = 2 * node_id + 2

                nodes_treeids.append(tree_id)
                nodes_nodeids.append(node_id)
                nodes_featureids.append(feature_idx)
                nodes_values.append(threshold)
                nodes_hitrates.append(1.0)
                nodes_modes.append(b"BRANCH_GT")
                nodes_truenodeids.append(left_id)
                nodes_falsenodeids.append(right_id)
                nodes_missing_value_tracks_true.append(0)

    attrs = {
        "nodes_treeids": Attribute(nodes_treeids, "INTS"),
        "nodes_nodeids": Attribute(nodes_nodeids, "INTS"),
        "nodes_featureids": Attribute(nodes_featureids, "INTS"),
        "nodes_values": Attribute(nodes_values, "FLOATS"),
        "nodes_hitrates": Attribute(nodes_hitrates, "FLOATS"),
        "nodes_modes": Attribute(nodes_modes, "STRINGS"),
        "nodes_truenodeids": Attribute(nodes_truenodeids, "INTS"),
        "nodes_falsenodeids": Attribute(nodes_falsenodeids, "INTS"),
        "nodes_missing_value_tracks_true": Attribute(nodes_missing_value_tracks_true, "INTS"),
        "target_treeids": Attribute(target_treeids, "INTS"),
        "target_nodeids": Attribute(target_nodeids, "INTS"),
        "target_ids": Attribute(target_ids, "INTS"),
        "target_weights": Attribute(target_weights, "FLOATS"),
    }

    if objective in ("Logloss", "CrossEntropy", "MultiClass"):
        attrs["classlabels_ints"] = Attribute([0, 1], "INTS")
        attrs["post_transform"] = Attribute(b"NONE", "STRING")
        out_label = ValueInfo("label", DType.INT64, ["batch_size"])
        out_prob = ValueInfo("probabilities", DType.FLOAT32, ["batch_size", 2])
        graph.outputs.extend([out_label, out_prob])

        node = Node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["label", "probabilities"],
            attributes=attrs,
            domain="ai.onnx.ml",
        )
        graph.nodes.append(node)

    else:
        attrs["n_targets"] = Attribute(1, "INT")
        attrs["post_transform"] = Attribute(b"NONE", "STRING")
        out_pred = ValueInfo("Y", DType.FLOAT32, ["batch_size", 1])
        graph.outputs.append(out_pred)

        node = Node(
            op_type="TreeEnsembleRegressor",
            inputs=["X"],
            outputs=["Y"],
            attributes=attrs,
            domain="ai.onnx.ml",
        )
        graph.nodes.append(node)

    return graph
