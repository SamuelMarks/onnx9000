import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass, IdentityEliminationPass


def test_cf_subgraph_recurse():
    g = Graph("Main")
    sub = Graph("Sub")
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    n1 = Node("Identity", ["A"], ["B"])
    sub.nodes.append(n1)

    n2 = Node("If", ["cond"], ["Y"])
    n2.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n2)

    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert not changed  # CF fold shouldn't fold Identity, IdentityElimination does.

    from onnx9000.optimizer.simplifier.passes.dce import IdentityEliminationPass

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_dce_loop_run_once_local_changed_twice():
    g = Graph("TestRunTwice")
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

    cf = ControlFlowFoldingPass()
    run_count = 0

    def mock_run_once(g):
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            return True
        return False

    cf._run_once = mock_run_once
    cf.run(g)


def test_cf_folding_with_subgraph_changed_trigger():
    # Force sub_changed = True, to test local_changed = True and changed = True
    g = Graph("TestSubTrigger")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c
    g.tensors["cond"].is_initializer = True
    g.outputs = ["Y"]

    sub = Graph("Sub")
    sub.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    sub.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    sub.outputs = ["Y"]

    n_drop = Node("Dropout", ["X"], ["Y"])
    sub.nodes.append(n_drop)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    n_if.attributes["else"] = Attribute("else", "GRAPH", Graph("E"))
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_dce_sub_changed_local_changed():
    g = Graph("TestSubTrig")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c

    sub = Graph("Sub")
    # node that triggers IdentityEliminationPass
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_cf_folding_with_subgraph_initializers():
    g = Graph("TestSubInit")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    t_c.is_initializer = True
    g.tensors["cond"] = t_c
    g.outputs = ["Y"]

    sub = Graph("Sub")
    # Subgraph initializer
    t_init = Tensor("A_init", (), DType.FLOAT32)
    t_init.is_initializer = True
    sub.tensors["A_init"] = t_init
    sub.initializers.append("A_init")

    sub.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    sub.outputs = [ValueInfo("Y", (), DType.FLOAT32)]

    n = Node("Abs", ["A_init"], ["Y"])
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then_branch"] = Attribute("then_branch", "GRAPH", sub)
    n_if.attributes["else_branch"] = Attribute("else_branch", "GRAPH", Graph("Else"))
    g.nodes.append(n_if)

    cf = ControlFlowFoldingPass()
    changed = cf.run(g)
    assert changed


def test_dce_sub_changed_local_changed2_1():
    g = Graph("TestSubTrig")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c

    sub = Graph("Sub")
    # A graph with nothing but an identity, we will change it manually to trigger recursion changed
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    run_count = 0
    # Monkey patch _run_once on the parent pass to trigger a change in the first pass
    orig_run_once = ie._run_once

    def mock_run_once(graph):
        nonlocal run_count
        if graph.name == "Sub" and run_count == 0:
            run_count += 1
            return True  # simulate change
        return orig_run_once(graph)

    ie._run_once = mock_run_once

    changed = ie.run(g)
    assert changed


def test_dce_sub_changed_local_changed2_2():
    g = Graph("TestSubTrig")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c

    sub = Graph("Sub")
    # A graph with nothing but an identity, we will change it manually to trigger recursion changed
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    run_count = 0
    # Monkey patch _run_once on the parent pass to trigger a change in the first pass
    orig_run_once = ie._run_once

    def mock_run_once(graph):
        nonlocal run_count
        if graph.name == "Sub" and run_count == 0:
            run_count += 1
            return True  # simulate change
        return orig_run_once(graph)

    ie._run_once = mock_run_once

    changed = ie.run(g)
    assert changed


def test_dce_sub_changed_local_changed2():
    g = Graph("TestSubTrig")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c

    sub = Graph("Sub")
    # A graph with nothing but an identity, we will change it manually to trigger recursion changed
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    run_count = 0
    # Monkey patch _run_once on the parent pass to trigger a change in the first pass
    orig_run_once = ie._run_once

    def mock_run_once(graph):
        nonlocal run_count
        if graph.name == "Sub" and run_count == 0:
            run_count += 1
            return True  # simulate change
        return orig_run_once(graph)

    ie._run_once = mock_run_once

    changed = ie.run(g)
    assert changed
