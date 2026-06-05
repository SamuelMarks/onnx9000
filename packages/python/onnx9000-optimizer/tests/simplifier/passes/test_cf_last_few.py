"""Tests for cf last few."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_last_few():
    """Docstring for D103."""
    cf = ConstantFoldingPass()

    g = Graph("G")
    g.inputs.append(ValueInfo("X", (2, 2), DType.FLOAT32))

    t = Tensor("t", (1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32).tobytes())
    t.is_initializer = True
    g.tensors["t"] = t
    g.initializers.append("t")

    # 423: new_node is None in partial_fold
    cf._partial_fold = lambda n, k: (None, True)
    n1 = Node("Abs", ["t"], ["out"])
    g.nodes.append(n1)

    # 424: new_node IS NOT None
    n1b = Node("Abs", ["t"], ["out1b"])
    g.nodes.append(n1b)

    def mock_partial(n, k):
        """Mock partial."""
        if n.name == "Abs" and n.outputs == ["out1b"]:
            return (Node("Identity", ["t"], ["out1b"]), True)
        return (None, True)

    cf._partial_fold = mock_partial

    # 432: Identity evaluate
    n2 = Node("Identity", ["t"], ["out2"])
    g.nodes.append(n2)

    cf._run_once(g)

    # 307: hasattr ndim
    g3 = Graph("G3")

    class MockVal:
        """Mock val."""

        ndim = 1
        shape = (1,)

        def __array__(self):
            """Array."""
            return np.array([1.0], dtype=np.float32)

    g3.nodes.append(Node("Constant", [], ["out3"], {"value": MockVal()}))
    cf._run_once(g3)

    # 370: float without nan/inf
    cf._evaluate_node = lambda *args, **kwargs: 1.0
    g4 = Graph("G4")
    g4.tensors["t"] = t
    g4.initializers.append("t")
    g4.nodes.append(Node("Abs", ["t"], ["out"]))
    cf._run_once(g4)
