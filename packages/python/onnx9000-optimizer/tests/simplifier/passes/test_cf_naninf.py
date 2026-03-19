import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_mock_nan_types():
    cf = ConstantFoldingPass()

    g = Graph("G1")
    t = Tensor("t", (), DType.FLOAT32)
    t.data = np.array(1.0, dtype=np.float32)
    t.is_initializer = True
    g.tensors["t"] = t
    g.initializers.append("t")
    g.nodes.append(Node("Abs", ["t"], ["out"]))

    # Mock _evaluate_node to return a raw python float
    cf._evaluate_node = lambda *args, **kwargs: float("inf")
    cf._run_once(g)  # Should hit float instance

    # Mock _evaluate_node to return a python list with a float
    cf._evaluate_node = lambda *args, **kwargs: [float("nan")]
    g2 = Graph("G2")
    g2.tensors["t"] = t
    g2.initializers.append("t")
    g2.nodes.append(Node("Abs", ["t"], ["out"]))
    cf._run_once(g2)  # Should hit list instance
