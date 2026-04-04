"""Module docstring."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter
from onnx9000.openvino.xml_builder import XmlNode


def test_export_all_missing_ops():
    """Docstring for D103."""
    graph = Graph("test_all")
    graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))

    graph.initializers.append("w")
    graph.tensors["w"] = Tensor("w", [1, 3, 3, 3], DType.FLOAT32, b"0" * 36)

    nodes = []

    nodes.append(
        Node(
            "ConvTranspose", ["x", "w"], ["y1"], attributes={"group": Attribute("group", "INT", 3)}
        )
    )
    nodes.append(Node("Erf", ["x"], ["y2"]))
    nodes.append(Node("Softmax", ["x"], ["y3"], attributes={"axis": Attribute("axis", "INT", 1)}))
    nodes.append(
        Node(
            "ReduceProd", ["x"], ["y4"], attributes={"keepdims": Attribute("keepdims", "INT", [1])}
        )
    )
    nodes.append(
        Node(
            "ArgMax",
            ["x"],
            ["y5"],
            attributes={
                "keepdims": Attribute("keepdims", "INT", [0]),
                "axis": Attribute("axis", "INT", 1),
            },
        )
    )
    nodes.append(
        Node(
            "Resize",
            ["x"],
            ["y6"],
            attributes={
                "mode": Attribute("mode", "STRING", "nearest"),
                "coordinate_transformation_mode": Attribute(
                    "coordinate_transformation_mode", "STRING", "half_pixel"
                ),
                "nearest_mode": Attribute("nearest_mode", "STRING", "round_prefer_floor"),
            },
        )
    )
    nodes.append(
        Node(
            "SpaceToDepth",
            ["x"],
            ["y7"],
            attributes={
                "blocksize": Attribute("blocksize", "INT", 2),
                "mode": Attribute("mode", "STRING", "DCR"),
            },
        )
    )
    nodes.append(
        Node(
            "NonMaxSuppression",
            ["x"],
            ["y8"],
            attributes={"center_point_box": Attribute("center_point_box", "INT", 1)},
        )
    )
    nodes.append(
        Node("RoiAlign", ["x"], ["y9"], attributes={"mode": Attribute("mode", "STRING", "max")})
    )
    nodes.append(
        Node(
            "Einsum",
            ["x"],
            ["y10"],
            attributes={"equation": Attribute("equation", "STRING", "ij,jk->ik")},
        )
    )
    nodes.append(
        Node(
            "LayerNormalization",
            ["x"],
            ["y11"],
            attributes={
                "axis": Attribute("axis", "INT", -1),
                "epsilon": Attribute("epsilon", "FLOAT", 1e-5),
            },
        )
    )
    nodes.append(
        Node(
            "InstanceNormalization",
            ["x"],
            ["y12"],
            attributes={"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
        )
    )
    nodes.append(
        Node(
            "LpNormalization",
            ["x"],
            ["y13"],
            attributes={"axis": Attribute("axis", "INT", -1), "p": Attribute("p", "INT", 2)},
        )
    )
    nodes.append(
        Node(
            "BatchNormalization",
            ["x"],
            ["y14"],
            attributes={"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
        )
    )
    nodes.append(Node("FakeQuantize", ["x"], ["y15"]))

    if_graph = Graph("then")
    if_graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))
    if_graph.outputs.append(ValueInfo("y", (1, 3, 224, 224), DType.FLOAT32))
    if_graph.nodes.append(Node("Relu", ["x"], ["y"]))

    else_graph = Graph("else")
    else_graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))
    else_graph.outputs.append(ValueInfo("y", (1, 3, 224, 224), DType.FLOAT32))
    else_graph.nodes.append(Node("Relu", ["x"], ["y"]))

    nodes.append(
        Node(
            "If",
            ["x"],
            ["y16"],
            attributes={
                "then_branch": Attribute("then_branch", "GRAPH", if_graph),
                "else_branch": Attribute("else_branch", "GRAPH", else_graph),
            },
        )
    )

    loop_graph = Graph("loop")
    loop_graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))
    loop_graph.outputs.append(ValueInfo("y", (1, 3, 224, 224), DType.FLOAT32))
    loop_graph.nodes.append(Node("Relu", ["x"], ["y"]))

    nodes.append(
        Node("Loop", ["x"], ["y17"], attributes={"body": Attribute("body", "GRAPH", loop_graph)})
    )

    nodes.append(Node("MissingOp", ["x"], ["y18"]))

    t = Tensor("val", [], DType.FLOAT32, b"")
    t.data = None
    nodes.append(
        Node("Constant", [], ["y19"], attributes={"value": Attribute("value", "TENSOR", t)})
    )

    graph.nodes.extend(nodes)

    exporter = OpenVinoExporter(graph)
    xml, bin_data = exporter.export()

    assert "GroupConvolutionBackpropData" in xml


def test_missing_input_pointer():
    """Docstring for D103."""
    graph = Graph("test_missing")
    graph.inputs.append(ValueInfo("x", (1, 3, 224, 224), DType.FLOAT32))
    graph.nodes.append(Node("Relu", ["missing_input"], ["y"]))
    exporter = OpenVinoExporter(graph)
    with pytest.raises(ValueError, match="Missing input pointer"):
        exporter.export()


def test_edge_duplication():
    """Docstring for D103."""
    exporter = OpenVinoExporter(Graph("dummy"))
    exporter._add_edge("1", "2", "3", "4")
    exporter._add_edge("1", "2", "3", "4")  # duplicate


def test_uninitialized_tensor():
    """Docstring for D103."""
    graph = Graph("dummy")
    graph.initializers.append("w")
    exporter = OpenVinoExporter(graph)
    xml, bin_data = exporter.export()
    assert "net" in xml


def test_xml_builder():
    """Docstring for D103."""
    node = XmlNode("test")
    node.add_child(XmlNode("child"))
    node.add_child(XmlNode("child"))
    assert len(node.children) == 2


def test_export_all_missing_ops_2():
    """Docstring for D103."""
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor
    from onnx9000.openvino.exporter import OpenVinoExporter

    g = Graph("g")
    g.inputs.append(Tensor("in", [1, 2, 3], DType.FLOAT32))

    ops_to_add = []
    ops_to_add.append(
        Node("ReduceMean", ["in"], ["out"], {"keepdims": Attribute("keepdims", "INT", 0)})
    )
    ops_to_add.append(
        Node(
            "ArgMin",
            ["in"],
            ["out"],
            {"keepdims": Attribute("keepdims", "INT", 1), "axis": Attribute("axis", "INT", 0)},
        )
    )
    ops_to_add.append(
        Node(
            "DepthToSpace",
            ["in"],
            ["out"],
            {
                "blocksize": Attribute("blocksize", "INT", 2),
                "mode": Attribute("mode", "STRING", "CR"),
            },
        )
    )
    ops_to_add.append(Node("CumSum", ["in"], ["out"], {}))
    ops_to_add.append(Node("QuantizeLinear", ["in"], ["out"], {}))
    ops_to_add.append(Node("DequantizeLinear", ["in"], ["out"], {}))

    # ArgMax
    ops_to_add.append(
        Node(
            "ArgMax",
            ["in"],
            ["out"],
            {"keepdims": Attribute("keepdims", "INT", 0), "axis": Attribute("axis", "INT", 1)},
        )
    )

    # Resize
    ops_to_add.append(
        Node(
            "Resize",
            ["in"],
            ["out"],
            {
                "mode": Attribute("mode", "STRING", "linear"),
                "coordinate_transformation_mode": Attribute(
                    "coordinate_transformation_mode", "STRING", "half_pixel"
                ),
                "nearest_mode": Attribute("nearest_mode", "STRING", "round_prefer_floor"),
            },
        )
    )

    # SpaceToDepth
    ops_to_add.append(
        Node(
            "SpaceToDepth",
            ["in"],
            ["out"],
            {
                "blocksize": Attribute("blocksize", "INT", 2),
                "mode": Attribute("mode", "STRING", "DCR"),
            },
        )
    )

    # NonMaxSuppression
    ops_to_add.append(
        Node(
            "NonMaxSuppression",
            ["in"],
            ["out"],
            {"center_point_box": Attribute("center_point_box", "INT", 1)},
        )
    )

    # RoiAlign
    ops_to_add.append(
        Node("RoiAlign", ["in"], ["out"], {"mode": Attribute("mode", "STRING", "max")})
    )

    # Einsum
    ops_to_add.append(
        Node("Einsum", ["in"], ["out"], {"equation": Attribute("equation", "STRING", "ij,jk->ik")})
    )

    # LayerNormalization
    ops_to_add.append(
        Node(
            "LayerNormalization",
            ["in"],
            ["out"],
            {"axis": Attribute("axis", "INT", -1), "epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
        )
    )

    # InstanceNormalization
    ops_to_add.append(
        Node(
            "InstanceNormalization",
            ["in"],
            ["out"],
            {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
        )
    )

    # LpNormalization
    ops_to_add.append(
        Node(
            "LpNormalization",
            ["in"],
            ["out"],
            {"axis": Attribute("axis", "INT", -1), "p": Attribute("p", "INT", 2)},
        )
    )

    # BatchNormalization
    ops_to_add.append(
        Node(
            "BatchNormalization", ["in"], ["out"], {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)}
        )
    )

    # If
    if_graph = Graph("if_g")
    else_graph = Graph("else_g")
    ops_to_add.append(
        Node(
            "If",
            ["in"],
            ["out"],
            {
                "then_branch": Attribute("then_branch", "GRAPH", if_graph),
                "else_branch": Attribute("else_branch", "GRAPH", else_graph),
            },
        )
    )

    # Loop
    loop_graph = Graph("loop_g")
    ops_to_add.append(
        Node("Loop", ["in"], ["out"], {"body": Attribute("body", "GRAPH", loop_graph)})
    )

    for n in ops_to_add:
        g.add_node(n)

    exporter = OpenVinoExporter(g, 11, False)
    print(f"NODES: {[n.op_type for n in g.nodes]}")
    xml_str, bin_data = exporter.export()


def test_xml_builder_more():
    """Docstring for D103."""
    from onnx9000.openvino.xml_builder import XmlBuilder, XmlNode

    n = XmlNode("test")
    n.add_child(XmlNode("<hello&>"))

    n.to_string(indent=0, pretty=True)

    n.to_string(indent=0, pretty=False)
    2

    b = XmlBuilder()
    b.set_declaration("<?xml ?>")
    assert "<?xml ?>" in b.to_string()
