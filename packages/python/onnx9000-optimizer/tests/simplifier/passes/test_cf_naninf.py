"""Tests for constant folding with NaN and infinity values."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_mock_nan_types():
    """Test constant folding with mocked NaN and infinity outputs."""
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


def test_erf():
    import sys
    from unittest.mock import MagicMock

    mock_scipy = MagicMock()
    mock_scipy.erf = lambda x: x
    sys.modules["scipy.special"] = mock_scipy
    sys.modules["scipy"] = MagicMock()
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph, Node
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("g")
    g.tensors["in"] = Constant(
        "in",
        values=np.array([0.0, 1.0], dtype=np.float32).tobytes(),
        dtype=DType.FLOAT32,
        shape=(2,),
    )
    g.initializers.append("in")
    g.add_node(Node("Erf", ["in"], ["out"]))
    cf = ConstantFoldingPass()
    cf._run_once(g)

    print(g.tensors)


def test_erf_inspect():
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant
    from onnx9000.optimizer.simplifier.passes.constant_folding import (
        _tensor_to_numpy,
    )

    t = Constant(
        "in",
        values=np.array([0.0, 1.0], dtype=np.float32).tobytes(),
        dtype=DType.FLOAT32,
        shape=(2,),
    )
    res = _tensor_to_numpy(t)
    print(f"Numpy: {res}")


def test_erf_inspect_exc():
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant

    t = Constant(
        "in",
        values=np.array([0.0, 1.0], dtype=np.float32).tobytes(),
        dtype=DType.FLOAT32,
        shape=(2,),
    )

    dtype_mapping = {DType.FLOAT32: np.float32}
    np_dtype = dtype_mapping.get(t.dtype)
    shape_list = [d.value if hasattr(d, "value") else d for d in t.shape]
    res = np.frombuffer(t.data, dtype=np_dtype).reshape(shape_list)
    print(f"RES: {res}")
