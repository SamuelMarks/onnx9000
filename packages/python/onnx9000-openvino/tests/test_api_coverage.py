import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, ValueInfo
from onnx9000.openvino.api import export_model


def test_api_export_model():
    graph = Graph("test_params")
    graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))
    node = Node("Relu", inputs=["x"], outputs=["y"])
    graph.nodes.append(node)

    xml, bin_data = export_model(graph, precision="fp16", clamp_dynamic=True)
    assert "<net" in xml
