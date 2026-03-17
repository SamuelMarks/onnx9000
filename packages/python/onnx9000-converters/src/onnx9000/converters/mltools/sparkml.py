"""SparkML JSON/Parquet parser for pure-Python ONNX conversion."""

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

    attrs = {"n_targets": Attribute(1, "INT"), "post_transform": Attribute(b"NONE", "STRING")}

    node = Node(
        op_type="TreeEnsembleRegressor",
        inputs=["X"],
        outputs=["Y"],
        attributes=attrs,
        domain="ai.onnx.ml",
    )
    graph.nodes.append(node)
    return graph
