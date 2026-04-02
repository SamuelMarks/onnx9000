"""Module providing functionality for test_dce_cov7."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor


def test_string_fallback():
    """Test string fallback."""
    from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass

    g = Graph("Str")
    g.outputs = ["old"]
    c = ControlFlowFoldingPass()
    c._rewire(g, "old", "new")
    assert g.outputs[0] == "new"


def test_dce_rewire_string_literal_exact():
    """Test dce rewire string literal exact."""
    g = Graph("TestRewireStr")
    g.outputs = ["old_out"]  # Use string instead of ValueInfo object

    from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass

    cf = ControlFlowFoldingPass()
    cf._rewire(g, "old_out", "new_out")

    assert g.outputs[0] == "new_out"


def test_cf_folding_with_subgraph_changed_trigger_cf():
    """Test that subgraph changes correctly trigger parent graph constant folding."""
    # Force sub_changed = True, to test local_changed = True and changed = True for ControlFlowFoldingPass
    g = Graph("TestSubTrigCF")
    g.inputs = ["cond"]
    t_c = Tensor("cond", (), DType.BOOL)
    t_c.data = np.array([True])
    g.tensors["cond"] = t_c

    sub = Graph("Sub")
    n = Node("Identity", ["A"], ["B"])
    sub.tensors["A"] = Tensor("A", (), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (), DType.FLOAT32)
    sub.nodes.append(n)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)

    from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass

    cf = ControlFlowFoldingPass()
    run_count = 0
    orig_run = cf.run

    def mock_run(graph):
        """Mocked run method to simulate subgraph changes."""
        nonlocal run_count
        if graph.name == "Sub" and run_count == 0:
            run_count += 1
            return True
        return orig_run(graph)

    cf.run = mock_run

    changed = cf.run(g)
    assert changed


def test_cf_rewire_inputs_name():
    """Provides functional implementation."""
    # Hit line 893: node.inputs[i] = new_name
    """Test cf rewire inputs name."""
    g = Graph("TestRewireInputsCF")
    g.inputs = ["X"]

    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["old_out"] = Tensor("old_out", (), DType.FLOAT32)
    g.tensors["new_out"] = Tensor("new_out", (), DType.FLOAT32)
    g.tensors["final_out"] = Tensor("final_out", (), DType.FLOAT32)

    # Node using old_out as input!
    n2 = Node("Relu", ["old_out"], ["final_out"])
    g.nodes.append(n2)

    from onnx9000.optimizer.simplifier.passes.dce import ControlFlowFoldingPass

    cf = ControlFlowFoldingPass()
    cf._rewire(g, "old_out", "X")
    assert n2.inputs[0] == "X"
