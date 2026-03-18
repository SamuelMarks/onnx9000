import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo, Attribute
from onnx9000.core.dtypes import DType
from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass


def test_if_folding():
    g = Graph("TestIf")
    g.inputs = ["cond_true", "cond_false", "in"]
    g.outputs = ["Y1", "Y2"]

    t_true = Tensor("cond_true", (1,), DType.BOOL)
    t_true.data = np.array([True])
    t_true.is_initializer = True
    g.tensors["cond_true"] = t_true

    t_false = Tensor("cond_false", (1,), DType.BOOL)
    t_false.data = np.array([False])
    t_false.is_initializer = True
    g.tensors["cond_false"] = t_false

    g.tensors["in"] = Tensor("in", (1,), DType.FLOAT32)

    then_g = Graph("Then")
    then_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))
    then_g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    n_then = Node("Abs", inputs=["in"], outputs=["out1"])
    then_g.nodes.append(n_then)

    else_g = Graph("Else")
    else_g.outputs.append(ValueInfo("out2", (1,), DType.FLOAT32))
    else_g.tensors["out2"] = Tensor("out2", (1,), DType.FLOAT32)
    n_else = Node("Neg", inputs=["in"], outputs=["out2"])
    else_g.nodes.append(n_else)

    n_if1 = Node("If", inputs=["cond_true"], outputs=["Y1"])
    n_if1.attributes["then_branch"] = Attribute("then_branch", None, then_g)
    n_if1.attributes["else_branch"] = Attribute("else_branch", None, else_g)
    g.nodes.append(n_if1)

    n_if2 = Node("If", inputs=["cond_false"], outputs=["Y2"])
    n_if2.attributes["then_branch"] = Attribute("then_branch", None, then_g)
    n_if2.attributes["else_branch"] = Attribute("else_branch", None, else_g)
    g.nodes.append(n_if2)

    cf_pass = ControlFlowFoldingPass()
    changed = cf_pass.run(g)
    assert changed

    ops = [n.op_type for n in g.nodes]
    assert "Abs" in ops  # From folded true
    assert "Neg" in ops  # From folded false
    assert "If" not in ops


def test_loop_zero_trip_folding():
    g = Graph("TestLoopZero")
    t_m = Tensor("M", (1,), DType.INT64)
    t_m.data = np.array([0])
    t_m.is_initializer = True
    g.tensors["M"] = t_m

    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.data = np.array([True])
    g.tensors["cond"] = t_cond

    g.tensors["in1"] = Tensor("in1", (1,), DType.FLOAT32)

    body_g = Graph("Body")
    body_g.nodes.append(Node("Abs", inputs=["in1"], outputs=["out1"]))
    body_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))

    n_loop = Node("Loop", inputs=["M", "cond", "in1"], outputs=["v_out_loop", "scan_out"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)

    cf_pass = ControlFlowFoldingPass()
    changed = cf_pass.run(g)
    assert changed
    ops = [n.op_type for n in g.nodes]
    assert "SequenceEmpty" in ops
    assert "Loop" not in ops


def test_loop_cond_false_folding():
    g = Graph("TestLoopCond")
    g.tensors["M"] = Tensor("M", (1,), DType.INT64)  # Unknown

    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.data = np.array([False])
    t_cond.is_initializer = True
    g.tensors["cond"] = t_cond

    g.tensors["in1"] = Tensor("in1", (1,), DType.FLOAT32)

    body_g = Graph("Body")
    body_g.nodes.append(Node("Abs", inputs=["in1"], outputs=["out1"]))
    body_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))

    n_loop = Node("Loop", inputs=["M", "cond", "in1"], outputs=["v_out_loop", "scan_out"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)

    cf_pass = ControlFlowFoldingPass()
    changed = cf_pass.run(g)
    assert changed
    ops = [n.op_type for n in g.nodes]
    assert "SequenceEmpty" in ops
    assert "Loop" not in ops


def test_loop_unroll_small():
    g = Graph("TestLoopUnroll")

    t_m = Tensor("M", (1,), DType.INT64)
    t_m.data = np.array([2])
    t_m.is_initializer = True
    g.tensors["M"] = t_m

    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.data = np.array([True])
    t_cond.is_initializer = True
    g.tensors["cond"] = t_cond

    g.tensors["v_in"] = Tensor("v_in", (1,), DType.FLOAT32)

    body_g = Graph("Body")
    body_g.inputs = ["iter", "cond", "v_in"]
    body_g.outputs.append(ValueInfo("cond_out", (1,), DType.BOOL))
    body_g.outputs.append(ValueInfo("v_out", (1,), DType.FLOAT32))
    body_g.outputs.append(ValueInfo("scan_out", (1,), DType.FLOAT32))

    n_abs = Node("Abs", inputs=["v_in"], outputs=["v_out"])
    n_neg = Node("Neg", inputs=["v_in"], outputs=["scan_out"])
    n_cond = Node("Identity", inputs=["cond"], outputs=["cond_out"])
    body_g.nodes.extend([n_abs, n_neg, n_cond])

    n_loop = Node("Loop", inputs=["M", "cond", "v_in"], outputs=["v_final", "scan_final"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)

    cf_pass = ControlFlowFoldingPass()
    changed = cf_pass.run(g)

    assert changed
    ops = [n.op_type for n in g.nodes]
    assert "Loop" not in ops
    assert ops.count("Abs") == 3
    assert ops.count("Neg") == 3
