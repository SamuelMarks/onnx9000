"""Tests for export."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter


def test_export_parameters():
    """Docstring for D103."""
    graph = Graph("test_params")
    graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))

    node = Node("Relu", inputs=["x"], outputs=["y"])
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="Parameter"' in xml_str
    assert 'name="x"' in xml_str
    assert 'element_type="f32"' in xml_str
    assert "<dim>224</dim>" in xml_str


def test_export_add():
    """Docstring for D103."""
    graph = Graph("test_add")
    graph.inputs.append(ValueInfo("A", (1, 3, 224, 224), DType.FLOAT32))
    graph.inputs.append(ValueInfo("B", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("C", (1, 3, 224, 224), DType.FLOAT32))
    node = Node(op_type="Add", inputs=["A", "B"], outputs=["C"])
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="Add"' in xml_str
    assert 'type="Parameter"' in xml_str
    assert 'type="Result"' in xml_str


def test_export_matmul():
    """Docstring for D103."""
    graph = Graph("test_matmul")
    graph.inputs.append(ValueInfo("A", (1, 3, 224, 224), DType.FLOAT32))
    graph.inputs.append(ValueInfo("B", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("C", (1, 3, 224, 224), DType.FLOAT32))

    from onnx9000.core.ir import Attribute

    node = Node(
        op_type="MatMul",
        inputs=["A", "B"],
        outputs=["C"],
        attributes={"transA": Attribute("transA", value=1), "transB": Attribute("transB", value=0)},
    )
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="MatMul"' in xml_str
    assert 'transpose_a="true"' in xml_str
    assert 'transpose_b="false"' in xml_str


def test_export_conv():
    """Docstring for D103."""
    graph = Graph("test_conv")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    graph.inputs.append(ValueInfo("W", (16, 3, 3, 3), DType.FLOAT32))
    graph.outputs.append(ValueInfo("Y", (1, 16, 222, 222), DType.FLOAT32))

    node = Node(
        op_type="Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        attributes={
            "strides": Attribute("strides", value=[1, 1]),
            "pads": Attribute("pads", value=[0, 0, 0, 0]),
        },
    )
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="Convolution"' in xml_str
    assert 'strides="1,1"' in xml_str
    assert 'pads_begin="0,0"' in xml_str
    assert 'pads_end="0,0"' in xml_str


def test_export_maxpool():
    """Docstring for D103."""
    graph = Graph("test_maxpool")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("Y", (1, 3, 112, 112), DType.FLOAT32))

    node = Node(
        op_type="MaxPool",
        inputs=["X"],
        outputs=["Y"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", value=[2, 2]),
            "strides": Attribute("strides", value=[2, 2]),
            "pads": Attribute("pads", value=[0, 0, 0, 0]),
        },
    )
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="MaxPool"' in xml_str
    assert 'kernel="2,2"' in xml_str
    assert 'strides="2,2"' in xml_str


def test_export_activations():
    """Docstring for D103."""
    graph = Graph("test_act")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("Y", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("Z", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("W", (1, 3, 224, 224), DType.FLOAT32))

    graph.nodes.append(Node(op_type="Relu", inputs=["X"], outputs=["Y"]))
    graph.nodes.append(Node(op_type="Sigmoid", inputs=["Y"], outputs=["Z"]))
    graph.nodes.append(
        Node(
            op_type="Softmax",
            inputs=["Z"],
            outputs=["W"],
            attributes={"axis": Attribute("axis", value=1)},
        )
    )

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="ReLU"' in xml_str
    assert 'type="Sigmoid"' in xml_str
    assert 'type="SoftMax"' in xml_str
    assert 'axis="1"' in xml_str


def test_export_fakequantize():
    """Docstring for D103."""
    graph = Graph("test_fq")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    graph.outputs.append(ValueInfo("Y", (1, 3, 224, 224), DType.FLOAT32))

    node = Node("QuantizeLinear", inputs=["X"], outputs=["Y"])
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    assert 'type="FakeQuantize"' in xml_str
    assert 'levels="256"' in xml_str
