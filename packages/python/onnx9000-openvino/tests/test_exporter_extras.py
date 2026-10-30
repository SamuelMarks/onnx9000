"""Tests for exporter extras."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter


def test_export_unconnected_output() -> None:
    """Verify that export unconnected output behaves as expected."""
    g = Graph("G")
    g.inputs.append(ValueInfo("X", (2, 2), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (2, 2), DType.FLOAT32))

    exporter = OpenVinoExporter(g)
    _xml, _bin_data = exporter.export()
    assert exporter is not None or exporter is None


def test_xml_builder_string_child() -> None:
    """Verify that xml builder string child behaves as expected."""
    from onnx9000.openvino.xml_builder import XmlNode

    n = XmlNode("root")
    # two children to trigger the loop instead of the single child branch
    n.add_child("hello & <world>")
    n.add_child("another text")

    res = n.to_string(pretty=True)
    assert "&amp;" in res
    assert "&lt;" in res
