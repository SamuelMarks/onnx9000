import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.core.dtypes import DType
from onnx9000.optimizer.simplifier.passes.dce import IdentityEliminationPass


def test_chained_expand():
    g = Graph("TestExpand")
    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["Shape1"] = Tensor("Shape1", (2,), DType.INT64)
    g.tensors["Shape2"] = Tensor("Shape2", (2,), DType.INT64)
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    g.tensors["Z"] = Tensor("Z", (2,), DType.FLOAT32)

    n1 = Node("Expand", inputs=["X", "Shape1"], outputs=["Y"])
    n2 = Node("Expand", inputs=["Y", "Shape2"], outputs=["Z"])
    g.nodes.extend([n1, n2])

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    assert n2.inputs[0] == "X"


def test_slice_dynamic_steps():
    g = Graph("TestSliceDynamic")
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

    g.tensors["steps"] = Tensor("steps", (1,), DType.INT64)  # Unknown steps!
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)

    n1 = Node("Slice", inputs=["X", "starts", "ends", "axes", "steps"], outputs=["Y"])
    g.nodes.append(n1)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert not changed  # dynamic steps, should not be eliminated


def test_reduce_scalar():
    g = Graph("TestReduce")
    g.tensors["X"] = Tensor("X", (), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)
    g.tensors["Z"] = Tensor("Z", (), DType.FLOAT32)

    n1 = Node("ReduceSum", inputs=["X"], outputs=["Y"])
    n2 = Node("ReduceMean", inputs=["Y"], outputs=["Z"])
    g.nodes.extend([n1, n2])

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    ops = [n.op_type for n in g.nodes]
    assert "ReduceSum" not in ops
    assert "ReduceMean" not in ops
