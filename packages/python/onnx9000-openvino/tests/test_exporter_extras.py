"""Tests for exporter extras."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter


def test_export_unconnected_output():
    """Docstring for D103."""
    g = Graph("G")
    g.inputs.append(ValueInfo("X", (2, 2), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (2, 2), DType.FLOAT32))

    exporter = OpenVinoExporter(g)
    xml, bin_data = exporter.export()


def test_xml_builder_string_child():
    """Docstring for D103."""
    from onnx9000.openvino.xml_builder import XmlBuilder, XmlNode

    n = XmlNode("root")
    # two children to trigger the loop instead of the single child branch
    n.add_child("hello & <world>")
    n.add_child("another text")

    res = n.to_string(pretty=True)
    assert "&amp;" in res
    assert "&lt;" in res
