"""Tests for dead code elimination within subgraphs."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.dce import DCEPass


def test_dce_subgraphs():
    """Test that dead nodes inside subgraphs (e.g., If branches) are correctly pruned."""
    g = Graph("Main")
    g.outputs = ["Y"]
    g.tensors["cond"] = Tensor("cond", (1,), DType.BOOL)
    g.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)

    sub = Graph("Sub")
    sub.outputs = ["SubY"]
    sub.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    sub.tensors["SubY"] = Tensor("SubY", (1,), DType.FLOAT32)
    sub.tensors["DeadSub"] = Tensor("DeadSub", (1,), DType.FLOAT32)

    sub.nodes.append(Node("Relu", ["X"], ["SubY"]))
    sub.nodes.append(Node("Relu", ["X"], ["DeadSub"]))  # Should be pruned

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then_branch"] = Attribute("then_branch", None, sub)
    g.nodes.append(n_if)

    dce = DCEPass()
    changed = dce.run(g)

    assert changed
    assert len(sub.nodes) == 1
    assert "DeadSub" not in [n.outputs[0] for n in sub.nodes]
