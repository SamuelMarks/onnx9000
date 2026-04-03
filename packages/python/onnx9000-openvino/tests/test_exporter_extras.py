import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, ValueInfo, Node, Tensor
from onnx9000.openvino.exporter import OpenVinoExporter


def test_export_unconnected_output():
    g = Graph("G")
    g.inputs.append(ValueInfo("X", (2, 2), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (2, 2), DType.FLOAT32))

    exporter = OpenVinoExporter(g)
    xml, bin_data = exporter.export()
