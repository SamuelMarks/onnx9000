import struct

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter


def test_export_unsupported_dtype():
    g = Graph("g")
    g.inputs.append(ValueInfo("x", (1,), DType.FLOAT16))
    exporter = OpenVinoExporter(g)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        exporter._map_dtype("UNSUPPORTED_DTYPE")


def test_clamp_dynamic():
    g = Graph("g")
    g.inputs.append(ValueInfo("x", ["batch", 3, -1], DType.FLOAT32))
    g.nodes.append(Node("Relu", ["x"], ["y"]))
    g.outputs.append(ValueInfo("y", ["batch", 3, -1], DType.FLOAT32))
    exporter = OpenVinoExporter(g, clamp_dynamic=True)
    xml, bin_ = exporter.export()
    assert "<dim>1</dim>" in xml


def test_dynamic_const_dtypes():
    g = Graph("g")
    exporter = OpenVinoExporter(g)

    # i64
    _, p = exporter._emit_dynamic_const("i64", [1], [1], DType.INT64)
    # i32
    _, p = exporter._emit_dynamic_const("i32", [1], [1], DType.INT32)
    # f32
    _, p = exporter._emit_dynamic_const("f32", [1.0], [1], DType.FLOAT32)

    with pytest.raises(ValueError):
        exporter._emit_dynamic_const("err", [1], [1], DType.INT8)


def test_conv_nodes():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    g.inputs.append(ValueInfo("W", (16, 3, 3, 3), DType.FLOAT32))
    g.inputs.append(ValueInfo("B", (16,), DType.FLOAT32))

    n1 = Node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        attributes={
            "group": Attribute("group", "INT", 2),
            "strides": Attribute("strides", "INTS", [2, 2]),
            "dilations": Attribute("dilations", "INTS", [1, 1]),
            "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
            "output_padding": Attribute("output_padding", "INTS", [0, 0]),
            "auto_pad": Attribute("auto_pad", "STRING", "SAME_UPPER"),
        },
    )

    n2 = Node(
        "ConvTranspose",
        inputs=["X", "W"],
        outputs=["Y2"],
        attributes={"pads": Attribute("pads", "INTS", [1, 1])},
    )

    n3 = Node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y3"],
        attributes={
            "pads": Attribute("pads", "INTS", [1, 1]),
            "auto_pad": Attribute("auto_pad", "STRING", "VALID"),
        },
    )

    g.nodes = [n1, n2, n3]
    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "GroupConvolution" in xml
    assert "ConvolutionBackpropData" in xml


def test_pool_nodes():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))
    n1 = Node(
        "MaxPool",
        inputs=["X"],
        outputs=["Y"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", "INTS", [3, 3]),
            "strides": Attribute("strides", "INTS", [2, 2]),
            "pads": Attribute("pads", "INTS", [1, 1, 1, 1]),
            "auto_pad": Attribute("auto_pad", "STRING", "SAME_UPPER"),
        },
    )
    n2 = Node(
        "AveragePool",
        inputs=["X"],
        outputs=["Y2"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", "INTS", [3, 3]),
            "pads": Attribute("pads", "INTS", [1, 1]),
            "count_include_pad": Attribute("count_include_pad", "INT", 1),
        },
    )
    n3 = Node(
        "AveragePool",
        inputs=["X"],
        outputs=["Y3"],
        attributes={"count_include_pad": Attribute("count_include_pad", "INT", 0)},
    )
    g.nodes = [n1, n2, n3]
    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "MaxPool" in xml
    assert "AvgPool" in xml


def test_gelu_softmax_concat_split():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))
    g.nodes.append(
        Node("Gelu", ["X"], ["Y1"], {"approximate": Attribute("approximate", "STRING", "tanh")})
    )
    g.nodes.append(
        Node("Gelu", ["X"], ["Y2"], {"approximate": Attribute("approximate", "STRING", "none")})
    )
    g.nodes.append(Node("Gelu", ["X"], ["Y3"]))
    g.nodes.append(Node("Softmax", ["X"], ["Y4"], {"axis": Attribute("axis", "INT", 1)}))
    g.nodes.append(Node("Concat", ["X", "X"], ["Y5"], {"axis": Attribute("axis", "INT", 1)}))
    g.nodes.append(Node("Split", ["X"], ["Y6", "Y7"], {"axis": Attribute("axis", "INT", 1)}))
    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "tanh" in xml


def test_pad_op():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))
    g.inputs.append(ValueInfo("Pads", (4,), DType.INT64))

    g.nodes.append(
        Node(
            "Pad",
            ["X"],
            ["Y1"],
            {
                "mode": Attribute("mode", "STRING", "reflect"),
                "pads": Attribute("pads", "INTS", [0, 1, 0, 1]),
                "value": Attribute("value", "FLOAT", 1.5),
            },
        )
    )

    g.nodes.append(
        Node("Pad", ["X", "Pads"], ["Y2"], {"mode": Attribute("mode", "STRING", "constant")})
    )

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "Pad" in xml


def test_gather_slice_reduce_argmax():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(
        Node(
            "Gather",
            ["X"],
            ["Y1"],
            {"batch_dims": Attribute("batch_dims", "INT", 1), "axis": Attribute("axis", "INT", 0)},
        )
    )
    g.nodes.append(Node("Gather", ["X", "X"], ["Y2"], {"axis": Attribute("axis", "INT", 1)}))

    g.nodes.append(Node("Slice", ["X"], ["Y3"]))

    g.nodes.append(
        Node(
            "ReduceMean",
            ["X"],
            ["Y4"],
            {"keepdims": Attribute("keepdims", "INT", 1), "axes": Attribute("axes", "INTS", [1])},
        )
    )
    g.nodes.append(Node("ReduceMax", ["X"], ["Y5"], {"keepdims": Attribute("keepdims", "INT", 0)}))

    g.nodes.append(
        Node(
            "ArgMax",
            ["X"],
            ["Y6"],
            {"keepdims": Attribute("keepdims", "INT", 1), "axis": Attribute("axis", "INT", 1)},
        )
    )
    g.nodes.append(Node("ArgMin", ["X"], ["Y7"], {"keepdims": Attribute("keepdims", "INT", 0)}))

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "Gather" in xml
    assert "StridedSlice" in xml


def test_resize_space_nms_roi():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(
        Node(
            "Resize",
            ["X"],
            ["Y1"],
            {
                "mode": Attribute("mode", "STRING", "linear"),
                "coordinate_transformation_mode": Attribute(
                    "coordinate_transformation_mode", "STRING", "pytorch_half_pixel"
                ),
                "nearest_mode": Attribute("nearest_mode", "STRING", "floor"),
            },
        )
    )

    g.nodes.append(
        Node(
            "SpaceToDepth",
            ["X"],
            ["Y2"],
            {
                "blocksize": Attribute("blocksize", "INT", 2),
                "mode": Attribute("mode", "STRING", "DCR"),
            },
        )
    )

    g.nodes.append(
        Node(
            "NonMaxSuppression",
            ["X"],
            ["Y3"],
            {"center_point_box": Attribute("center_point_box", "INT", 1)},
        )
    )
    g.nodes.append(Node("NonMaxSuppression", ["X"], ["Y4"]))

    g.nodes.append(Node("RoiAlign", ["X"], ["Y5"], {"mode": Attribute("mode", "STRING", "max")}))
    g.nodes.append(Node("RoiAlign", ["X"], ["Y6"]))

    g.nodes.append(Node("QuantizeLinear", ["X"], ["Y7"]))
    g.nodes.append(
        Node("Einsum", ["X"], ["Y8"], {"equation": Attribute("equation", "STRING", "ij,jk->ik")})
    )

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "Interpolate" in xml


def test_norms():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(
        Node(
            "LayerNormalization",
            ["X"],
            ["Y1"],
            {"axis": Attribute("axis", "INT", -1), "epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
        )
    )
    g.nodes.append(
        Node(
            "InstanceNormalization", ["X"], ["Y2"], {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)}
        )
    )
    g.nodes.append(
        Node(
            "LpNormalization",
            ["X"],
            ["Y3"],
            {"axis": Attribute("axis", "INT", -1), "p": Attribute("p", "INT", 2)},
        )
    )
    g.nodes.append(
        Node("BatchNormalization", ["X"], ["Y4"], {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)})
    )

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "MVN" in xml


def test_dropout_cast_gridsample():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(Node("Dropout", ["X"], ["Y1"]))
    g.nodes.append(Node("Cast", ["X"], ["Y2"], {"to": Attribute("to", "INT", DType.INT64.value)}))

    g.nodes.append(
        Node(
            "GridSample",
            ["X"],
            ["Y3"],
            {
                "mode": Attribute("mode", "STRING", "bilinear"),
                "padding_mode": Attribute("padding_mode", "STRING", "zeros"),
                "align_corners": Attribute("align_corners", "INT", 1),
            },
        )
    )

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "Convert" in xml


def test_size_flatten_transpose():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(Node("Size", ["X"], ["Y1"]))
    g.nodes.append(Node("Flatten", ["X"], ["Y2"]))
    g.nodes.append(Node("Transpose", ["X"], ["Y3"], {"perm": Attribute("perm", "INTS", [1, 0])}))

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "ShapeOf" in xml


def test_gather_elements_constantofshape():
    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1, 3), DType.FLOAT32))

    g.nodes.append(Node("GatherElements", ["X"], ["Y1"], {"axis": Attribute("axis", "INT", 1)}))

    c = Constant("c", shape=(1,), dtype=DType.FLOAT32)
    c.data = struct.pack("<f", 3.14)
    g.nodes.append(
        Node("ConstantOfShape", ["X"], ["Y2"], {"value": Attribute("value", "TENSOR", c)})
    )

    c2 = Constant("c2", shape=(1,), dtype=DType.INT64)
    c2.data = struct.pack("<q", 42)
    g.nodes.append(
        Node("ConstantOfShape", ["X"], ["Y3"], {"value": Attribute("value", "TENSOR", c2)})
    )

    c3 = Constant("c3", shape=(1,), dtype=DType.INT32)
    c3.data = struct.pack("<i", 43)
    g.nodes.append(
        Node("ConstantOfShape", ["X"], ["Y4"], {"value": Attribute("value", "TENSOR", c3)})
    )

    g.nodes.append(Node("ConstantOfShape", ["X"], ["Y5"]))

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "GatherElements" in xml


def test_control_flow():
    sub_g1 = Graph("sub1")
    sub_g1.inputs.append(ValueInfo("a", (1,), DType.BOOL))
    sub_g1.nodes.append(Node("Relu", ["a"], ["b"]))

    sub_g2 = Graph("sub2")
    sub_g2.inputs.append(ValueInfo("c", (1,), DType.BOOL))
    sub_g2.nodes.append(Node("Sigmoid", ["c"], ["d"]))

    g = Graph("g")
    g.inputs.append(ValueInfo("X", (1,), DType.BOOL))
    g.nodes.append(
        Node(
            "If",
            ["X"],
            ["Y1"],
            {
                "then_branch": Attribute("then_branch", "GRAPH", sub_g1),
                "else_branch": Attribute("else_branch", "GRAPH", sub_g2),
            },
        )
    )

    g.nodes.append(Node("Loop", ["X"], ["Y2"], {"body": Attribute("body", "GRAPH", sub_g1)}))

    g.nodes.append(Node("Scan", ["X"], ["Y3"], {"body": Attribute("body", "GRAPH", sub_g1)}))

    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "If" in xml
    assert "TensorIterator" in xml


def test_mat_gemm():
    g = Graph("g")
    g.inputs.append(ValueInfo("A", (2, 2), DType.FLOAT32))
    g.inputs.append(ValueInfo("B", (2, 2), DType.FLOAT32))
    g.inputs.append(ValueInfo("C", (2,), DType.FLOAT32))

    g.nodes.append(
        Node(
            "MatMul",
            ["A", "B"],
            ["Y1"],
            {"transA": Attribute("transA", "INT", 1), "transB": Attribute("transB", "INT", 1)},
        )
    )
    g.nodes.append(
        Node(
            "Gemm",
            ["A", "B", "C"],
            ["Y2"],
            {"transA": Attribute("transA", "INT", 0), "transB": Attribute("transB", "INT", 0)},
        )
    )
    exporter = OpenVinoExporter(g)
    xml, bin_ = exporter.export()
    assert "transpose_a" in xml
