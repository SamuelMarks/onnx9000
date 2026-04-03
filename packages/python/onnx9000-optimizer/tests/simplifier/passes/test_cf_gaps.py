"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/passes/test_cf_gaps.py."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_shape_gaps():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    cf = ConstantFoldingPass()
    g = Graph("G")
    g.tensors["X"] = Tensor("X", None, DType.FLOAT32)  # missing shape
    g.inputs.append(ValueInfo("X", None, DType.FLOAT32))
    g.nodes.append(Node("Shape", ["X"], ["out"]))
    cf._run_once(g)  # Hit `if x_shape is None: return False`

    g2 = Graph("G2")
    t_val = Tensor("t_val", (1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32).tobytes())
    g2.nodes.append(Node("Constant", [], ["out"], {"value": t_val}))

    class MockAttr:
        attr_type = "TENSOR"
        value = b"test"

    g2.nodes.append(Node("Constant", [], ["out2"], {"value": MockAttr()}))

    from unittest.mock import patch

    with patch(
        "onnx9000.core.parser.core.parse_tensor_proto",
        return_value=Tensor(
            "t", (1,), DType.FLOAT32, data=np.array([2.0], dtype=np.float32).tobytes()
        ),
    ):
        cf._run_once(g2)  # Hit `elif hasattr(val, "attr_type")`


def test_cf_shape_dynamic():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass
    from onnx9000.core.ir import DynamicDim

    cf = ConstantFoldingPass()
    g = Graph("G")
    g.inputs.append(ValueInfo("X", (DynamicDim("A"), 2), DType.FLOAT32))
    g.nodes.append(Node("Shape", ["X"], ["out"]))
    cf._run_once(g)


def test_cf_gaps_misc():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    cf = ConstantFoldingPass()

    # Missing BitwiseShift OTHER
    assert (
        cf._evaluate_node("BitShift", [np.array([4]), np.array([1])], {"direction": "OTHER"})
        is None
    )

    # Missing Conv
    assert cf._evaluate_node("Conv", [], {}) is None
    """Test shape folding fallback."""
    g = Graph("TestShapeFallback")
    g.inputs = [ValueInfo("X", (2, 3), DType.FLOAT32)]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 3), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2,), DType.INT64)
    n = Node("Shape", ["X"], ["Y"])
    g.nodes.append(n)
    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert changed
    assert g.nodes[0].op_type == "Constant"


def test_shape_folding_tensor_dict():
    """Test shape folding tensor dict."""
    g = Graph("TestShapeTensor")
    g.tensors["X"] = Tensor("X", (4, 5), DType.FLOAT32)
    g.outputs = ["Y"]
    g.tensors["Y"] = Tensor("Y", (2,), DType.INT64)
    n = Node("Shape", ["X"], ["Y"])
    g.nodes.append(n)
    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert changed
    assert g.nodes[0].op_type == "Constant"


def test_custom_op_warning():
    """Test custom op warning."""
    g = Graph("TestCustom")
    t1 = Tensor("C1", (1,), DType.FLOAT32)
    t1.data = np.array([1.0], dtype=np.float32)
    t1.is_initializer = True
    g.tensors["C1"] = t1
    n = Node("MyCustomOp", ["C1"], ["Y"])
    n.domain = "my.domain"
    g.nodes.append(n)
    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert not changed


def test_cf_nan_inf_types():
    """Test cf nan inf types."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("TestNanInfTypes")
    t1 = Tensor("t1", (1,), DType.FLOAT32)
    t1.data = np.array([float("inf")])
    t1.is_initializer = True
    g.tensors["t1"] = t1
    n1 = Node("Identity", ["t1"], ["out1"])
    g.nodes.append(n1)
    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert not changed
    t2 = Tensor("t2", (1,), DType.STRING)
    t2.data = np.array(["hello"])
    t2.is_initializer = True
    g.tensors["t2"] = t2
    n2 = Node("Identity", ["t2"], ["out2"])
    g.nodes.append(n2)
    cf._run_once(g)

    class MockNode:
        """MockNode implementation."""

        op_type = "MockList"
        inputs = ["t2"]
        outputs = ["out3"]
        attributes = {}
        domain = "ai.onnx"
        name = "mock"

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            ConstantFoldingPass, "_evaluate_node", lambda *args: [np.array([1.0], dtype=np.float32)]
        )
        g.nodes.append(Node("Split", ["t2"], ["out3", "out4"]))
        cf._run_once(g)


def test_cf_div_by_zero():
    """Test cf div by zero."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node
    n = Node("Div", [], [])
    with pytest.raises(ValueError, match="Div by zero"):
        _evaluate_node(n.op_type, [np.array([1.0]), np.array([0.0])], n.attributes)


def test_cf_concat_scalars():
    """Test cf concat scalars."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node
    n = Node("Concat", [], [])
    n.attributes["axis"] = Attribute("axis", value=0)
    out = _evaluate_node(n.op_type, [1.0, np.array([2.0])], n.attributes)
    assert out.shape == (2,)


def test_cf_bn_slice():
    """Test cf bn slice."""
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node
    n = Node("BatchNormalization", [], [])
    X = np.ones((1, 2, 2, 2))
    scale = np.ones((2,))
    B = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))
    out = _evaluate_node(n.op_type, [X, scale, B, mean, var], n.attributes)
    assert out.shape == (1, 2, 2, 2)
    X1d = np.ones((1, 2))
    out1d = _evaluate_node(n.op_type, [X1d, scale, B, mean, var], n.attributes)
    assert out1d.shape == (1, 2)
