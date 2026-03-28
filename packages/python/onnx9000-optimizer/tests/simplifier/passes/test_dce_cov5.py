"""Module providing functionality for test_dce_cov5."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import IdentityEliminationPass


def test_chained_expand():
    """Test chained expand."""
    g = Graph("TestChainedExpand")
    g.inputs = ["X"]
    g.outputs = ["Z"]

    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["S1"] = Tensor("S1", (1,), DType.INT64)
    g.tensors["S2"] = Tensor("S2", (1,), DType.INT64)
    g.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)
    g.tensors["Z"] = Tensor("Z", (1,), DType.FLOAT32)

    n1 = Node("Expand", ["X", "S1"], ["Y"])
    n2 = Node("Expand", ["Y", "S2"], ["Z"])
    g.nodes.extend([n1, n2])

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_reduce_scalar():
    """Test reduce scalar."""
    g = Graph("TestReduceScalar")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)  # Scalar shape!
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)

    n1 = Node("ReduceSum", ["X"], ["Y"])
    g.nodes.append(n1)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed


def test_rewire_out_string():
    """Test rewire out string."""
    g = Graph("TestRewireStr")
    g.inputs = ["X"]
    g.outputs = ["old_out"]  # Raw string in outputs
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["old_out"] = Tensor("old_out", (), DType.FLOAT32)

    n1 = Node("Identity", ["X"], ["old_out"])
    g.nodes.append(n1)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    assert g.outputs[0] == "X"
