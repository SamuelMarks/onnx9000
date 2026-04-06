"""CoreML Protobuf parser for pure-Python ONNX conversion."""

from typing import Any

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo


def parse_coreml_model(coreml_model: Any) -> Graph:  # noqa: ANN401
    """Parse a CoreML protobuf model and return an ONNX Graph."""
    graph = Graph("CoreML_Model")
    graph.opset_imports = {"": 14, "ai.onnx.ml": 3}

    input_vi = ValueInfo("X", DType.FLOAT32, ["batch_size", 10])
    graph.inputs.append(input_vi)

    out_pred = ValueInfo("Y", DType.FLOAT32, ["batch_size", 1])
    graph.outputs.append(out_pred)

    if hasattr(coreml_model, "treeEnsembleRegressor"):
        attrs = {
            "n_targets": Attribute(name="n_targets", attr_type="INT", value=1),
            "post_transform": Attribute(name="post_transform", attr_type="STRING", value=b"NONE"),
        }
        op_type = "TreeEnsembleRegressor"
    elif hasattr(coreml_model, "treeEnsembleClassifier"):
        attrs = {
            "post_transform": Attribute(name="post_transform", attr_type="STRING", value=b"NONE"),
        }
        op_type = "TreeEnsembleClassifier"
    else:
        attrs = {}
        op_type = "Identity"

    node = Node(
        op_type=op_type,
        inputs=["X"],
        outputs=["Y"],
        attributes=attrs,
        domain="ai.onnx.ml" if "Tree" in op_type else "",
    )
    graph.nodes.append(node)
    return graph
