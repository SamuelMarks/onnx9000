import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import DCEPass


def test_dce_run_once_missing():
    g = Graph("TestDCE")
    g.inputs = ["X", "UnusedIn"]
    g.outputs = ["Y"]
    g.value_info = [ValueInfo("UnusedVI", (), DType.FLOAT32)]
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    g.tensors["UnusedVI"] = Tensor("UnusedVI", (2,), DType.FLOAT32)

    n_used = Node("Relu", inputs=["X"], outputs=["Y"], name="n_used")
    n_dead = Node("Relu", inputs=["X"], outputs=["UnusedVI"], name="n_dead")
    g.nodes.extend([n_used, n_dead])

    dce = DCEPass(unused_inputs_to_prune=["UnusedIn"])
    changed = dce._run_once(g)
    assert changed
    assert len(g.nodes) == 1
    assert "UnusedIn" not in g.inputs
    assert len(g.value_info) == 0
