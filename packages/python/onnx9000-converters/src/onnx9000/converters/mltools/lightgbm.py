"""LightGBM JSON parser for pure-Python ONNX conversion."""

import json
from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_lightgbm_json(json_str: str) -> Graph:
    """Parse a LightGBM JSON dump and return an ONNX Graph."""
    data = json.loads(json_str)
    return parse_lightgbm_dict(data)


def parse_lightgbm_dict(data: dict[str, Any]) -> Graph:
    """Parse a LightGBM dict representation."""
    graph = Graph("LightGBM_Model")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    num_features = data.get("max_feature_idx", 0) + 1
    input_shape = ["batch_size", num_features]
    input_vi = ValueInfo("X", DType.FLOAT32, input_shape)
    graph.inputs.append(input_vi)

    objective = data.get("objective", "")

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

    for tree in data.get("tree_info", []):
        tree_id = tree["tree_index"]

        def traverse(node: dict[str, Any], node_id: int, tree_id: int = tree_id) -> int:
            """Executes the traverse operation."""
            if "leaf_value" in node:
                nodes_treeids.append(tree_id)
                nodes_nodeids.append(node_id)
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_hitrates.append(1.0)
                nodes_modes.append(b"LEAF")
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_value_tracks_true.append(0)

                target_treeids.append(tree_id)
                target_nodeids.append(node_id)
                target_ids.append(0)
                target_weights.append(node["leaf_value"])
                return node_id
            else:
                current_node_id = node_id
                left_id = traverse(node["left_child"], current_node_id * 2 + 1)
                right_id = traverse(node["right_child"], current_node_id * 2 + 2)

                nodes_treeids.append(tree_id)
                nodes_nodeids.append(current_node_id)
                nodes_featureids.append(node["split_feature"])
                nodes_values.append(float(node["threshold"]))
                nodes_hitrates.append(1.0)
                nodes_modes.append(b"BRANCH_LEQ")
                nodes_truenodeids.append(left_id)
                nodes_falsenodeids.append(right_id)
                missing = 1 if node.get("default_left", True) else 0
                nodes_missing_value_tracks_true.append(missing)
                return current_node_id

        traverse(tree["tree_structure"], 0)

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

    if "binary" in objective:
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
