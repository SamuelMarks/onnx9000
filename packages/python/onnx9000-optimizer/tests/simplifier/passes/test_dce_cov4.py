"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/passes/test_dce_cov4.py."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import (
    ControlFlowFoldingPass,
    DCEPass,
    IdentityEliminationPass,
)


def test_slice_dynamic_inputs_with_defaults():
    """Test slice dynamic inputs with defaults."""
    g = Graph("TestSliceDefaults")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    t_start = Tensor("starts", (1,), DType.INT64)
    t_start.data = np.array([0])
    t_start.is_initializer = True
    g.tensors["starts"] = t_start
    t_end = Tensor("ends", (1,), DType.INT64)
    t_end.data = np.array([9999999999999999])
    t_end.is_initializer = True
    g.tensors["ends"] = t_end
    t_axis = Tensor("axes", (1,), DType.INT64)
    t_axis.data = np.array([0])
    t_axis.is_initializer = True
    g.tensors["axes"] = t_axis
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes"], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_reduce_scalar_vi():
    """Test reduce scalar vi."""
    g = Graph("TestReduceVI")
    g.inputs = [ValueInfo("X", (), DType.FLOAT32)]
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    n1 = Node("ReduceSum", inputs=["X"], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_cf_scalars():
    """Test cf scalars."""
    g = Graph("TestIfScalar")
    t_cond = Tensor("cond", (), DType.BOOL)
    t_cond.data = True
    t_cond.is_initializer = True
    g.tensors["cond"] = t_cond
    g.tensors["out"] = Tensor("out", (1,), DType.FLOAT32)
    then_g = Graph("Then")
    then_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))
    then_g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    else_g = Graph("Else")
    else_g.outputs.append(ValueInfo("out2", (1,), DType.FLOAT32))
    else_g.tensors["out2"] = Tensor("out2", (1,), DType.FLOAT32)
    n_if = Node("If", inputs=["cond"], outputs=["out"])
    n_if.attributes["then_branch"] = Attribute("then", None, then_g)
    n_if.attributes["else_branch"] = Attribute("else", None, else_g)
    g.nodes.append(n_if)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_cf_loop_scalars():
    """Test cf loop scalars."""
    g = Graph("TestLoopScalar")
    t_m = Tensor("M", (), DType.INT64)
    t_m.data = 0
    t_m.is_initializer = True
    g.tensors["M"] = t_m
    t_cond = Tensor("cond", (), DType.BOOL)
    t_cond.data = False
    t_cond.is_initializer = True
    g.tensors["cond"] = t_cond
    body_g = Graph("Body")
    n_loop = Node("Loop", inputs=["M", "cond"], outputs=["out"])
    n_loop.attributes["body"] = Attribute("body", None, body_g)
    g.nodes.append(n_loop)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_slice_dynamic_steps_not_init():
    """Test slice dynamic steps not init."""
    g = Graph("TestSliceSteps")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    t_start = Tensor("starts", (1,), DType.INT64)
    t_start.data = np.array([0])
    t_start.is_initializer = True
    g.tensors["starts"] = t_start
    t_end = Tensor("ends", (1,), DType.INT64)
    t_end.data = np.array([9999])
    t_end.is_initializer = True
    g.tensors["ends"] = t_end
    t_axis = Tensor("axes", (1,), DType.INT64)
    t_axis.data = np.array([0])
    t_axis.is_initializer = True
    g.tensors["axes"] = t_axis
    g.tensors["steps"] = Tensor("steps", (1,), DType.INT64)
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes", "steps"], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert not changed


def test_cf_subgraph_tensors():
    """Test cf subgraph tensors."""
    g = Graph("TestIfSubG")
    g.inputs = ["cond"]
    g.outputs = ["Y"]
    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.data = np.array([True])
    g.tensors["cond"] = t_cond
    g.tensors["cond"].is_initializer = True
    then_g = Graph("Then")
    then_g.tensors["t1"] = Tensor("t1", (1,), DType.FLOAT32)
    then_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))
    then_g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    then_g.nodes.append(Node("Abs", inputs=["t1"], outputs=["out1"]))
    else_g = Graph("Else")
    else_g.outputs.append(ValueInfo("out2", (1,), DType.FLOAT32))
    else_g.tensors["out2"] = Tensor("out2", (1,), DType.FLOAT32)
    n_if = Node("If", inputs=["cond"], outputs=["Y"])
    n_if.attributes["then_branch"] = Attribute("then", None, then_g)
    n_if.attributes["else_branch"] = Attribute("else", None, else_g)
    g.nodes.append(n_if)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_cf_multi_run():
    """Test cf multi run."""
    g = Graph("TestMultiRun")
    cf = ControlFlowFoldingPass()
    assert not cf.run(g)


def test_dce_slice_dynamic_steps_string():
    """Test dce slice dynamic steps string."""
    g = Graph("TestSliceStepsStr")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    t_start = Tensor("starts", (1,), DType.INT64)
    t_start.data = np.array([0])
    t_start.is_initializer = True
    g.tensors["starts"] = t_start
    t_end = Tensor("ends", (1,), DType.INT64)
    t_end.data = np.array([9999999999999999])
    t_end.is_initializer = True
    g.tensors["ends"] = t_end
    t_axis = Tensor("axes", (1,), DType.INT64)
    t_axis.data = np.array([0])
    t_axis.is_initializer = True
    g.tensors["axes"] = t_axis
    g.tensors["steps"] = Tensor("steps", (1,), DType.INT64)
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes", "steps"], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert not changed


def test_dce_loop_run_once_local_changed():
    """Test dce loop run once local changed."""
    g = Graph("TestRunOnce")
    sub = Graph("Sub")
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)
    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    g.nodes.append(n_if)
    dce = DCEPass()
    run_count = 0

    def _run_once_mock(g):
        """Perform  run once mock operation."""
        nonlocal run_count
        run_count += 1
        if getattr(g, "name", "") == "Sub" and run_count < 3:
            return True
        return False

    dce._run_once = _run_once_mock
    changed = dce.run(g)
    assert changed


def test_dce_if_tensor_not_initializer():
    """Test dce if tensor not initializer."""
    g = Graph("TestIfTensors")
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["cond"].data = True
    g.tensors["cond"].is_initializer = True
    then_g = Graph("Then")
    then_g.tensors["t1"] = Tensor("t1", (), DType.FLOAT32)
    then_g.nodes.append(Node("Abs", ["t1"], ["out1"]))
    then_g.outputs.append("out1")
    else_g = Graph("Else")
    else_g.outputs.append("out2")
    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then_branch"] = Attribute("then_branch", "GRAPH", then_g)
    n_if.attributes["else_branch"] = Attribute("else_branch", "GRAPH", else_g)
    g.nodes.append(n_if)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_cf_folding_with_subgraph_tensors():
    """Test cf folding with subgraph tensors."""
    g = Graph("TestSubTensors")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c
    g.tensors["cond"].is_initializer = True
    g.outputs = ["Y"]
    then_g = Graph("Then")
    t_hidden = Tensor("hidden", (1,), DType.FLOAT32)
    then_g.tensors["hidden"] = t_hidden
    n_hidden = Node("Abs", ["in1"], ["hidden"])
    then_g.nodes.append(n_hidden)
    then_g.tensors["in1"] = Tensor("in1", (1,), DType.FLOAT32)
    then_g.outputs.append(ValueInfo("hidden", (1,), DType.FLOAT32))
    else_g = Graph("Else")
    else_g.outputs.append(ValueInfo("hidden2", (1,), DType.FLOAT32))
    else_g.tensors["hidden2"] = Tensor("hidden2", (1,), DType.FLOAT32)
    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then_branch"] = Attribute("then", "GRAPH", then_g)
    n_if.attributes["else_branch"] = Attribute("else", "GRAPH", else_g)
    g.nodes.append(n_if)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_slice_dynamic_steps_empty_string():
    """Test slice dynamic steps empty string."""
    g = Graph("TestSliceStepsEmptyStr")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    t_start = Tensor("starts", (1,), DType.INT64)
    t_start.data = np.array([0])
    t_start.is_initializer = True
    g.tensors["starts"] = t_start
    t_end = Tensor("ends", (1,), DType.INT64)
    t_end.data = np.array([9999999999999999])
    t_end.is_initializer = True
    g.tensors["ends"] = t_end
    t_axis = Tensor("axes", (1,), DType.INT64)
    t_axis.data = np.array([0])
    t_axis.is_initializer = True
    g.tensors["axes"] = t_axis
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes", ""], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_dce_rewire_out_is_string():
    """Test dce rewire out is string."""
    g = Graph("TestRewireStr")
    g.inputs = ["X"]
    g.outputs = ["old_out"]
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["old_out"] = Tensor("old_out", (), DType.FLOAT32)
    g.tensors["new_out"] = Tensor("new_out", (), DType.FLOAT32)
    n1 = Node("Identity", ["X"], ["old_out"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    assert g.outputs[0] == "X"


def test_slice_dynamic_steps_not_1():
    """Test slice dynamic steps not 1."""
    g = Graph("TestSliceStepsNot1")
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    t_start = Tensor("starts", (1,), DType.INT64)
    t_start.data = np.array([0])
    t_start.is_initializer = True
    g.tensors["starts"] = t_start
    t_end = Tensor("ends", (1,), DType.INT64)
    t_end.data = np.array([9999999999999999])
    t_end.is_initializer = True
    g.tensors["ends"] = t_end
    t_axis = Tensor("axes", (1,), DType.INT64)
    t_axis.data = np.array([0])
    t_axis.is_initializer = True
    g.tensors["axes"] = t_axis
    t_steps = Tensor("steps", (1,), DType.INT64)
    t_steps.data = np.array([2])
    t_steps.is_initializer = True
    g.tensors["steps"] = t_steps
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes", "steps"], outputs=["Y"])
    g.nodes.append(n1)
    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert not changed


def test_cf_subgraph_changed():
    """Test cf subgraph changed."""
    g = Graph("TestIfSubChanged")
    g.inputs = ["cond"]
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["cond"].data = np.array([True])
    g.tensors["cond"].is_initializer = True
    g.outputs = ["Y"]
    then_g = Graph("Then")
    then_g.tensors["t1"] = Tensor("t1", (1,), DType.FLOAT32)
    then_g.tensors["out1"] = Tensor("out1", (1,), DType.FLOAT32)
    then_g.outputs.append(ValueInfo("out1", (1,), DType.FLOAT32))
    then_g.nodes.append(Node("Dropout", inputs=["t1"], outputs=["out1"]))
    n_if = Node("If", inputs=["cond"], outputs=["Y"])
    n_if.attributes["then_branch"] = Attribute("then", "GRAPH", then_g)
    g.nodes.append(n_if)
    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed
