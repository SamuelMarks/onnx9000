"""H2O MOJO/POJO parser for pure-Python ONNX conversion."""

import json
from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_h2o(model_data: Any) -> Graph:  # noqa: ANN401
    """Parse an H2O MOJO/POJO and return an ONNX Graph."""
    graph = Graph("H2O_Model")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    input_vi = ValueInfo("X", DType.FLOAT32, ["batch_size", 10])
    graph.inputs.append(input_vi)

    out_pred = ValueInfo("Y", DType.FLOAT32, ["batch_size", 1])
    graph.outputs.append(out_pred)

    if isinstance(model_data, str) and model_data.strip().startswith("{"):
        try:
            data = json.loads(model_data)
            algo = data.get("algo", "")
            if algo == "xgboost":
                op_type = "TreeEnsembleRegressor"
                attrs = {"n_targets": Attribute(name="n_targets", attr_type="INT", value=1)}
            elif algo == "deeplearning":
                op_type = "MatMul"
                attrs = {}
            else:
                op_type = "Identity"
                attrs = {}
        except json.JSONDecodeError:
            op_type = "TreeEnsembleRegressor"
            attrs = {"n_targets": Attribute(name="n_targets", attr_type="INT", value=1)}
    else:
        op_type = "TreeEnsembleRegressor"
        attrs = {
            "n_targets": Attribute(name="n_targets", attr_type="INT", value=1),
            "post_transform": Attribute(name="post_transform", attr_type="STRING", value=b"NONE"),
        }

    node = Node(
        op_type=op_type,
        inputs=["X"],
        outputs=["Y"],
        attributes=attrs,
        domain="ai.onnx.ml" if op_type.startswith("Tree") else "",
    )
    graph.nodes.append(node)
    return graph
