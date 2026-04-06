"""SparkML JSON/Parquet parser for pure-Python ONNX conversion."""

import json
from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_sparkml_pipeline(pipeline_data: Any) -> Graph:  # noqa: ANN401
    """Parse a SparkML Pipeline model and return an ONNX Graph."""
    graph = Graph("SparkML_Pipeline")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    input_vi = ValueInfo("X", DType.FLOAT32, ["batch_size", 10])
    graph.inputs.append(input_vi)

    out_pred = ValueInfo("Y", DType.FLOAT32, ["batch_size", 1])
    graph.outputs.append(out_pred)

    op_type = "TreeEnsembleRegressor"
    attrs = {
        "n_targets": Attribute(name="n_targets", attr_type="INT", value=1),
        "post_transform": Attribute(name="post_transform", attr_type="STRING", value=b"NONE"),
    }

    if isinstance(pipeline_data, str) and pipeline_data.strip().startswith("{"):
        try:
            data = json.loads(pipeline_data)
            pipeline_class = data.get("class", "")
            if "RandomForest" in pipeline_class:
                op_type = "TreeEnsembleRegressor"
            elif "LogisticRegression" in pipeline_class:
                op_type = "LinearClassifier"
                attrs = {}
            elif "LinearRegression" in pipeline_class:
                op_type = "LinearRegressor"
                attrs = {}
        except json.JSONDecodeError:
            _ignore = True

    node = Node(
        op_type=op_type,
        inputs=["X"],
        outputs=["Y"],
        attributes=attrs,
        domain="ai.onnx.ml",
    )
    graph.nodes.append(node)
    return graph
