from onnx9000.script import op, Var, GraphBuilder
import pytest
import numpy as np


def test_op_namespace():
    with GraphBuilder() as builder:
        x = Var("x")
        y = Var("y")
        z = op.Add(x, y)
        assert isinstance(z, Var)
        w = op.Relu(z)
        assert isinstance(w, Var)
        assert len(builder.nodes) == 2
        assert builder.nodes[0].op_type == "Add"
        assert builder.nodes[1].op_type == "Relu"


def test_op_multi_output():
    with GraphBuilder() as builder:
        x = Var("x")
        val, idx = op.TopK(x, 5)
        assert isinstance(val, Var)
        assert isinstance(idx, Var)
        assert len(builder.nodes) == 2
        assert builder.nodes[1].op_type == "TopK"
        assert len(builder.nodes[1].outputs) == 2


def test_op_constant_cast():
    with GraphBuilder() as builder:
        x = Var("x")
        z = op.Add(x, 1)
        assert len(builder.nodes) == 2
        assert builder.nodes[0].op_type == "Constant"
        assert builder.nodes[1].op_type == "Add"


def test_op_constant_explicit():
    with GraphBuilder() as builder:
        c1 = op.Constant(1)
        c2 = op.Constant(1.5)
        c3 = op.Constant([1, 2])
        c4 = op.Constant([1.0, 2.0])
        c5 = op.Constant(np.array([1, 2], dtype=np.int32))
        with pytest.raises(ValueError):
            op.Constant("string")
        assert len(builder.nodes) == 5


def test_var_overloads():
    with GraphBuilder() as builder:
        x = Var("x")
        y = Var("y")
        _ = x + y
        _ = x - y
        _ = x * y
        _ = x / y
        _ = x**y
        _ = x @ y
        _ = x > y
        _ = x < y
        _ = x == y
        _ = x != y
        _ = x & y
        _ = x | y
        _ = x ^ y
        _ = ~x
        _ = x[0]
        _ = x[1:3]
        _ = 1 + x
        _ = 1 - x
        _ = 1 * x
        _ = 1 / x
        c = op.Concat([x, y], axis=0)
        assert len(builder.nodes) == 30
        ops = [n.op_type for n in builder.nodes]
        assert "Add" in ops
        assert "Sub" in ops
        assert "Mul" in ops
        assert "Div" in ops
        assert "Pow" in ops
        assert "MatMul" in ops
        assert "Greater" in ops
        assert "Less" in ops
        assert "Equal" in ops
        assert "NotEqual" in ops
        assert "BitwiseAnd" in ops
        assert "BitwiseOr" in ops
        assert "BitwiseXor" in ops
        assert "BitwiseNot" in ops
        assert "Gather" in ops
        assert "Slice" in ops


def test_op_control_flow():
    with GraphBuilder() as builder:
        cond = Var("cond")
        max_trip = Var("max_trip")
        if_out = op.If(cond, "then_graph", "else_graph")
        loop_out = op.Loop(max_trip, cond, "loop_body")
        scan_out = op.Scan("scan_body", 1)
        assert len(builder.nodes) == 3
        ops = [n.op_type for n in builder.nodes]
        assert ops == ["If", "Loop", "Scan"]
        assert builder.nodes[0].attributes["then_branch"] == "then_graph"
        assert builder.nodes[1].attributes["body"] == "loop_body"
        assert builder.nodes[2].attributes["num_scan_inputs"] == 1


def test_op_schema_validation():
    with GraphBuilder() as builder:
        x = Var("x")
        s = op.Squeeze(x, axes=[1])
        assert builder.nodes[1].op_type == "Squeeze"
        assert len(builder.nodes[1].inputs) == 2
        assert "axes" not in builder.nodes[1].attributes
        op.set_target_opset(10)
        with pytest.raises(ValueError, match="requires opset"):
            op.Add(x, x)
        op.set_target_opset(18)


def test_var_rename():
    x = Var("x")
    assert repr(x) == "Var(x)"
    x.rename("new_x")
    assert x.name == "new_x"
