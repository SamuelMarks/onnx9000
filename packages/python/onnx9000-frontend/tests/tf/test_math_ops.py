from onnx9000.frontend.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontend.tf.math_ops import MATH_OPS_MAPPING
from onnx9000.frontend.tf.parsers import TFNode


def test_math_ops_mapping() -> None:
    builder = TFToONNXGraphBuilder()
    node = TFNode("n1", "Add", inputs=["a", "b"])
    outs = MATH_OPS_MAPPING["Add"](builder, node)
    assert len(outs) == 1
    assert builder.graph.nodes[-1].op_type == "Add"
    node = TFNode("n2", "Abs", inputs=["x"])
    outs = MATH_OPS_MAPPING["Abs"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Abs"
    node = TFNode("n3", "FloorDiv", inputs=["a", "b"])
    outs = MATH_OPS_MAPPING["FloorDiv"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Floor"
    assert builder.graph.nodes[-2].op_type == "Div"
    node = TFNode("n4", "FloorMod", inputs=["a", "b"])
    outs = MATH_OPS_MAPPING["FloorMod"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Mod"
    assert builder.graph.nodes[-1].attributes["fmod"] == 0
    node = TFNode("n5", "Square", inputs=["a"])
    outs = MATH_OPS_MAPPING["Square"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Mul"
    assert builder.graph.nodes[-1].inputs == ["a", "a"]
    node = TFNode("n6", "Rsqrt", inputs=["a"])
    outs = MATH_OPS_MAPPING["Rsqrt"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Reciprocal"
    assert builder.graph.nodes[-2].op_type == "Sqrt"
    node = TFNode("n7", "Expm1", inputs=["a"])
    outs = MATH_OPS_MAPPING["Expm1"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Sub"
    assert builder.graph.nodes[-2].op_type == "Exp"
    node = TFNode("n8", "Log1p", inputs=["a"])
    outs = MATH_OPS_MAPPING["Log1p"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Log"
    assert builder.graph.nodes[-2].op_type == "Add"
    node = TFNode("n9", "Atan2", inputs=["a", "b"])
    outs = MATH_OPS_MAPPING["Atan2"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Custom_Atan2"
    node = TFNode("n10", "IsFinite", inputs=["a"])
    outs = MATH_OPS_MAPPING["IsFinite"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Not"
    assert builder.graph.nodes[-2].op_type == "Or"
    assert builder.graph.nodes[-3].op_type == "IsInf"
    assert builder.graph.nodes[-4].op_type == "IsNaN"
    node = TFNode("n11", "ComplexAbs", inputs=["a"])
    outs = MATH_OPS_MAPPING["ComplexAbs"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Custom_ComplexAbs"
    node = TFNode("n12", "Angle", inputs=["a"])
    outs = MATH_OPS_MAPPING["Angle"](builder, node)
    assert builder.graph.nodes[-1].op_type == "Custom_Angle"
    expected_ops = [
        "Add",
        "AddV2",
        "Sub",
        "Mul",
        "Div",
        "RealDiv",
        "TruncateDiv",
        "FloorDiv",
        "Mod",
        "FloorMod",
        "Abs",
        "Neg",
        "Sign",
        "Reciprocal",
        "Square",
        "Sqrt",
        "Rsqrt",
        "Exp",
        "Expm1",
        "Log",
        "Log1p",
        "Ceil",
        "Floor",
        "Round",
        "Maximum",
        "Minimum",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Atan2",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Erf",
        "IsNan",
        "IsInf",
        "IsFinite",
        "Pow",
        "ComplexAbs",
        "Angle",
    ]
    for op in expected_ops:
        assert op in MATH_OPS_MAPPING
