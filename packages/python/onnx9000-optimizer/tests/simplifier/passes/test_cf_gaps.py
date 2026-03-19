import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_shape_folding_fallback():
    g = Graph("TestShapeFallback")

    g.inputs = [ValueInfo("X", (2, 3), DType.FLOAT32)]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 3), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2,), DType.INT64)

    # We want X to not be an initializer, so all_known will be False
    # Then node.op_type == "Shape" triggers fallback
    n = Node("Shape", ["X"], ["Y"])
    g.nodes.append(n)

    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert changed
    assert g.nodes[0].op_type == "Constant"


def test_shape_folding_tensor_dict():
    g = Graph("TestShapeTensor")
    # Not in graph.inputs
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

    # Simulate a list result
    class MockNode:
        op_type = "MockList"
        inputs = ["t2"]
        outputs = ["out3"]
        attributes = {}
        domain = "ai.onnx"
        name = "mock"

    # We will trigger the list size calc path in the code directly using a fake pass
    # But wait, it's easier to mock evaluate_constant_node to return a list.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            ConstantFoldingPass, "_evaluate_node", lambda *args: [np.array([1.0], dtype=np.float32)]
        )
        g.nodes.append(Node("Split", ["t2"], ["out3", "out4"]))
        cf._run_once(g)


def test_cf_div_by_zero():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node

    n = Node("Div", [], [])
    with pytest.raises(ValueError, match="Div by zero"):
        _evaluate_node(n.op_type, [np.array([1.0]), np.array([0.0])], n.attributes)


def test_cf_concat_scalars():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node
    n = Node("Concat", [], [])
    n.attributes["axis"] = Attribute("axis", value=0)

    # One scalar float, one float array
    out = _evaluate_node(n.op_type, [1.0, np.array([2.0])], n.attributes)
    assert out.shape == (2,)


def test_cf_bn_slice():
    # BatchNormalization
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    _evaluate_node = ConstantFoldingPass()._evaluate_node
    n = Node("BatchNormalization", [], [])

    X = np.ones((1, 2, 2, 2))
    scale = np.ones((2,))
    B = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))

    # Test bn evaluation
    out = _evaluate_node(n.op_type, [X, scale, B, mean, var], n.attributes)
    assert out.shape == (1, 2, 2, 2)

    # Test bn 1d
    X1d = np.ones((1, 2))
    out1d = _evaluate_node(n.op_type, [X1d, scale, B, mean, var], n.attributes)
    assert out1d.shape == (1, 2)
