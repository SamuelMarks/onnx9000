import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.constant_folding import (
    ConstantFoldingPass,
    _evaluate_conv,
    _evaluate_pool,
    _numpy_to_tensor_proto,
    _tensor_to_numpy,
)


def test_evaluate_pool_max_avg():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

    # MaxPool
    kernel = [2, 2]
    strides = [2, 2]
    pads = [0, 0, 0, 0]
    out_max = _evaluate_pool(x, kernel, strides, pads, pool_mode="max", ceil_mode=0)
    assert out_max.shape == (1, 1, 2, 2)
    assert out_max[0, 0, 0, 0] == 5.0

    # AvgPool
    out_avg = _evaluate_pool(x, kernel, strides, pads, pool_mode="avg", ceil_mode=1)
    assert out_avg.shape == (1, 1, 2, 2)
    assert out_avg[0, 0, 0, 0] == 2.5

    # With pads
    pads = [1, 1, 1, 1]
    out_max_pad = _evaluate_pool(x, kernel, strides, pads, pool_mode="max", ceil_mode=0)
    assert out_max_pad.shape == (1, 1, 3, 3)

    # Without pads array
    out_max_no_pad = _evaluate_pool(x, kernel, strides, pads=None, pool_mode="max", ceil_mode=0)
    assert out_max_no_pad.shape == (1, 1, 2, 2)


def test_evaluate_conv():
    # 1D conv
    x_1d = np.ones((1, 2, 5), dtype=np.float32)
    w_1d = np.ones((4, 2, 3), dtype=np.float32)
    b_1d = np.ones((4,), dtype=np.float32)
    out_1d = _evaluate_conv(x_1d, w_1d, b_1d, strides=[1], pads=[0, 0], dilations=[1], group=1)
    assert out_1d.shape == (1, 4, 3)

    # 2D conv with pads, strides, dilations, bias
    x = np.ones((1, 2, 5, 5), dtype=np.float32)
    w = np.ones((4, 2, 3, 3), dtype=np.float32)
    b = np.ones((4,), dtype=np.float32)

    out = _evaluate_conv(x, w, b, strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    assert out.shape == (1, 4, 5, 5)

    # None defaults
    out_none = _evaluate_conv(x, w, None, strides=None, pads=None, dilations=None, group=1)
    assert out_none.shape == (1, 4, 3, 3)

    # Grouped conv
    x_g = np.ones((1, 4, 5, 5), dtype=np.float32)
    w_g = np.ones((4, 2, 3, 3), dtype=np.float32)
    out_g = _evaluate_conv(
        x_g, w_g, None, strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=2
    )
    assert out_g.shape == (1, 4, 3, 3)


def test_numpy_to_tensor_proto():
    t_f32 = _numpy_to_tensor_proto(np.array([1.0], dtype=np.float32), "test_f32")
    assert t_f32.name == "test_f32"

    _numpy_to_tensor_proto(np.array([1.0], dtype=np.float64))
    _numpy_to_tensor_proto(np.array([1.0], dtype=np.float16))
    _numpy_to_tensor_proto(np.array([1], dtype=np.int32))
    _numpy_to_tensor_proto(np.array([1], dtype=np.int64))
    _numpy_to_tensor_proto(np.array([1], dtype=np.int16))
    _numpy_to_tensor_proto(np.array([1], dtype=np.int8))
    _numpy_to_tensor_proto(np.array([1], dtype=np.uint8))
    _numpy_to_tensor_proto(np.array([1], dtype=np.uint16))
    _numpy_to_tensor_proto(np.array([1], dtype=np.uint32))
    _numpy_to_tensor_proto(np.array([1], dtype=np.uint64))
    _numpy_to_tensor_proto(np.array([True], dtype=np.bool_))

    # Int and float conversion
    t_int = _numpy_to_tensor_proto(5)
    assert len(t_int.dims) == 0
    t_float = _numpy_to_tensor_proto(5.0)
    assert len(t_float.dims) == 0

    # String fallback
    t_str = _numpy_to_tensor_proto(np.array(["test"], dtype="O"))
    assert t_str.data_type == 0  # UNDEFINED


def test_tensor_to_numpy():
    # Numpy data directly
    t1 = Tensor("t1", (1,), DType.FLOAT32)
    t1.data = np.array([1.0], dtype=np.float32)
    assert _tensor_to_numpy(t1) is t1.data

    # None data
    t2 = Tensor("t2", (1,), DType.FLOAT32)
    t2.data = None
    assert _tensor_to_numpy(t2) is None

    t_err = Tensor("t_err", (2,), DType.FLOAT32)
    t_err.data = b"abc"  # Invalid length for float32 array of shape 2
    assert _tensor_to_numpy(t_err) is None


def test_tensor_to_numpy_bfloat16():
    # BFloat16 fallback
    t3 = Tensor("t3", (1,), DType.BFLOAT16)
    t3.data = np.array([0x3F80], dtype=np.uint16).tobytes()  # roughly 1.0
    arr3 = _tensor_to_numpy(t3)
    assert arr3.dtype == np.float32
    assert arr3.shape == (1,)


def test_tensor_to_numpy_none_dtype():
    # Invalid dtype mapping
    t4 = Tensor("t4", (1,), DType.STRING)  # Not in mapping
    t4.data = b"abc"
    assert _tensor_to_numpy(t4) is None


def test_cf_run_once_shape():
    g = Graph("TestShapeCF")
    g.tensors["X"] = Tensor("X", (2, 3), DType.FLOAT32)
    from onnx9000.core.ir import ValueInfo

    g.inputs = [ValueInfo("X", (2, 3), DType.FLOAT32)]
    g.outputs = ["Y"]
    g.tensors["Y"] = Tensor("Y", (2,), DType.INT64)

    n_shape = Node("Shape", ["X"], ["Y"])
    g.nodes.append(n_shape)

    cf = ConstantFoldingPass()
    changed = cf.run(g)
    assert changed
    assert g.nodes[0].op_type == "Constant"


def test_cf_run_once_shape_dynamic():
    g = Graph("TestShapeDyn")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    from onnx9000.core.ir import DynamicDim

    g.tensors["X"] = Tensor("X", (2, DynamicDim("D")), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2,), DType.INT64)

    n_shape = Node("Shape", ["X"], ["Y"])
    g.nodes.append(n_shape)

    cf = ConstantFoldingPass()
    changed = cf.run(g)
    assert not changed  # Should not fold dynamic shape


def test_cf_run_once_nan_inf():
    g = Graph("TestNanInf")
    t_c1 = Tensor("C1", (1,), DType.FLOAT32)
    t_c1.data = np.array([1.0])
    t_c1.is_initializer = True
    g.tensors["C1"] = t_c1
    t_c2 = Tensor("C2", (1,), DType.FLOAT32)
    t_c2.data = np.array([0.0])
    t_c2.is_initializer = True
    g.tensors["C2"] = t_c2
    g.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)

    # 1.0 / 0.0 -> Inf
    n_div = Node("Div", ["C1", "C2"], ["Y"])
    g.nodes.append(n_div)

    cf = ConstantFoldingPass()
    changed = cf.run(g)
    assert not changed  # Should skip folding due to Inf


def test_cf_has_nan_inf_types():
    g = Graph("TestNanInfTypes2")
    t1 = Tensor("t1", (1,), DType.FLOAT32)
    t1.data = np.array([float("inf")])
    t1.is_initializer = True
    g.tensors["t1"] = t1

    n1 = Node("Identity", ["t1"], ["out1"])
    g.nodes.append(n1)

    cf = ConstantFoldingPass()
    changed = cf._run_once(g)
    assert not changed

    # Test scalar float inf
    t2 = Tensor("t2", (), DType.FLOAT32)
    t2.data = float("inf")
    t2.is_initializer = True
    g.tensors["t2"] = t2
    n2 = Node("Identity", ["t2"], ["out2"])
    g.nodes.append(n2)
    cf._run_once(g)

    # Test list result with float inf
    t3 = Tensor("t3", (), DType.FLOAT32)
    t3.data = float("inf")
    t3.is_initializer = True
    g.tensors["t3"] = t3

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            ConstantFoldingPass, "_evaluate_node", lambda *args: [np.array([1.0]), float("nan")]
        )
        g.nodes.append(Node("Split", ["t3"], ["out3", "out4"]))
        cf._run_once(g)


def test_cf_div_by_zero():
    n = Node("Div", [], [])
    with pytest.raises(ValueError, match="Div by zero"):
        ConstantFoldingPass()._evaluate_node(
            n.op_type, [np.array([1.0]), np.array([0.0])], n.attributes
        )


def test_cf_concat_scalars():
    n = Node("Concat", [], [])
    n.attributes["axis"] = Attribute("axis", value=0)

    # One scalar float, one float array
    out = ConstantFoldingPass()._evaluate_node(n.op_type, [1.0, np.array([2.0])], n.attributes)
    assert out.shape == (2,)


def test_cf_bn_slice():
    # BatchNormalization
    n = Node("BatchNormalization", [], [])

    X = np.ones((1, 2, 2, 2))
    scale = np.ones((2,))
    B = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))

    # Test bn evaluation
    out = ConstantFoldingPass()._evaluate_node(n.op_type, [X, scale, B, mean, var], n.attributes)
    assert out.shape == (1, 2, 2, 2)

    # Test bn 1d
    X1d = np.ones((1, 2))
    out1d = ConstantFoldingPass()._evaluate_node(
        n.op_type, [X1d, scale, B, mean, var], n.attributes
    )
    assert out1d.shape == (1, 2)


def test_cf_gather():
    g = Graph("TestGather")

    t_data = Tensor("data", (4,), DType.FLOAT32)
    t_data.data = np.array([1.0, 2.0, 3.0, 4.0])
    t_data.is_initializer = True
    g.tensors["data"] = t_data

    t_ind = Tensor("ind", (2,), DType.INT64)
    t_ind.data = np.array([0, 3])
    t_ind.is_initializer = True
    g.tensors["ind"] = t_ind

    g.tensors["out"] = Tensor("out", (2,), DType.FLOAT32)

    n = Node("Gather", ["data", "ind"], ["out"])
    g.nodes.append(n)

    cf = ConstantFoldingPass()
    out = cf._evaluate_node(n.op_type, [t_data.data, t_ind.data], n.attributes)
    assert out is not None


def test_cf_gather_nd():
    g = Graph("TestGatherND")

    t_data = Tensor("data", (2, 2), DType.FLOAT32)
    t_data.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    t_data.is_initializer = True
    g.tensors["data"] = t_data

    t_ind = Tensor("ind", (1, 2), DType.INT64)
    t_ind.data = np.array([[1, 1]])
    t_ind.is_initializer = True
    g.tensors["ind"] = t_ind

    g.tensors["out"] = Tensor("out", (1,), DType.FLOAT32)

    n = Node("GatherND", ["data", "ind"], ["out"])
    n.attributes["batch_dims"] = Attribute("batch_dims", "INT", 0)
    g.nodes.append(n)

    cf = ConstantFoldingPass()
    out = cf._evaluate_node(n.op_type, [t_data.data, t_ind.data], n.attributes)
    assert out is not None


def test_gathernd_batch_dims():
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[1, 1]])

    # Test branch where batch_dims is not an Attribute object
    cf = ConstantFoldingPass()
    out = cf._evaluate_node("GatherND", [data, indices], {"batch_dims": 0})
    assert out is not None


def test_cf_constant_attr_types():
    from onnx9000.core.ir import Attribute, Graph, Node
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("TestCFConstants")

    # Has ndim
    n1 = Node("Constant", [], ["out1"])

    class HasNdim:
        ndim = 1

    n1.attributes["value"] = HasNdim()

    # value_float
    n2 = Node("Constant", [], ["out2"])
    n2.attributes["value_float"] = [1.0, 2.0]

    # value_int
    n3 = Node("Constant", [], ["out3"])
    n3.attributes["value_int"] = [1, 2]

    g.nodes.extend([n1, n2, n3])

    # Just run once, they should populate known_values
    # We can't easily check known_values, but we can hit the lines.
    cf = ConstantFoldingPass()
    cf._run_once(g)

    # Test the recursive path hit via sub_changed = True
    g2 = Graph("Main")
    sub = Graph("Sub")
    sub.nodes.append(Node("Constant", [], ["out4"]))
    sub.nodes[0].attributes["value_int"] = [1]

    # We need something to fold in the subgraph so run_once returns True
    sub.tensors["t"] = Tensor("t", (), DType.FLOAT32)
    sub.tensors["t"].data = 1.0
    sub.tensors["t"].is_initializer = True
    sub.nodes.append(Node("Identity", ["t"], ["out5"]))

    n_if = Node("If", [], [])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g2.nodes.append(n_if)

    cf2 = ConstantFoldingPass()
    cf2.run(g2)


def test_cf_constant_attr_types():
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("TestCFConstants")

    # Has ndim
    n1 = Node("Constant", [], ["out1"])

    class HasNdim:
        ndim = 1

    n1.attributes["value"] = Attribute("value", "CUSTOM", value=HasNdim())

    # value_float
    n2 = Node("Constant", [], ["out2"])
    n2.attributes["value_float"] = Attribute("value_float", "FLOATS", value=[1.0, 2.0])

    # value_int
    n3 = Node("Constant", [], ["out3"])
    n3.attributes["value_int"] = Attribute("value_int", "INTS", value=[1, 2])

    g.nodes.extend([n1, n2, n3])

    cf = ConstantFoldingPass()
    cf._run_once(g)

    # Test the recursive path hit via sub_changed = True
    g2 = Graph("Main")
    sub = Graph("Sub")
    sub.nodes.append(Node("Constant", [], ["out4"]))
    sub.nodes[0].attributes["value_int"] = Attribute("value_int", "INTS", value=[1])

    sub.tensors["t"] = Tensor("t", (), DType.FLOAT32)
    sub.tensors["t"].data = 1.0
    sub.tensors["t"].is_initializer = True
    sub.nodes.append(Node("Identity", ["t"], ["out5"]))

    n_if = Node("If", [], [])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g2.nodes.append(n_if)

    cf2 = ConstantFoldingPass()
    cf2.run(g2)


def test_cf_tensor_proto():
    import unittest.mock

    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Attribute, Graph, Node, Tensor
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    g = Graph("TestTensorProto")
    n = Node("Constant", [], ["out"])

    class MockProto:
        pass

    n.attributes["value"] = Attribute("value", "TENSOR", value=MockProto())
    g.nodes.append(n)

    t = Tensor("t", (1,), DType.FLOAT32)
    t.data = np.array([1.0])

    with unittest.mock.patch("onnx9000.core.parser.core.parse_tensor_proto", return_value=t):
        cf = ConstantFoldingPass()
        cf._run_once(g)


def test_cf_scatternd():
    cf = ConstantFoldingPass()
    data = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1.0, 2.0], dtype=np.float32)

    # default none
    out = cf._evaluate_node("ScatterND", [data, indices, updates], {})
    assert out[0, 0] == 1.0
    assert out[1, 1] == 2.0

    # add
    out_add = cf._evaluate_node(
        "ScatterND", [data, indices, updates], {"reduction": Attribute("r", "STRING", "add")}
    )
    assert out_add[0, 0] == 1.0

    # mul
    data2 = np.ones((3, 3), dtype=np.float32) * 2
    out_mul = cf._evaluate_node(
        "ScatterND", [data2, indices, updates], {"reduction": Attribute("r", "STRING", "mul")}
    )
    assert out_mul[0, 0] == 2.0

    # max
    out_max = cf._evaluate_node(
        "ScatterND", [data2, indices, updates], {"reduction": Attribute("r", "STRING", "max")}
    )
    assert out_max[1, 1] == 2.0

    # min
    out_min = cf._evaluate_node(
        "ScatterND", [data2, indices, updates], {"reduction": Attribute("r", "STRING", "min")}
    )
    assert out_min[0, 0] == 1.0


def test_cf_sequence_ops():
    cf = ConstantFoldingPass()

    # SequenceConstruct
    t1 = np.array([1])
    t2 = np.array([2])
    out_seq = cf._evaluate_node("SequenceConstruct", [t1, t2], {})
    assert isinstance(out_seq, list)
    assert len(out_seq) == 2

    # SequenceAt
    idx = np.array([1])
    out_at = cf._evaluate_node("SequenceAt", [out_seq, idx], {})
    assert out_at[0] == 2

    # SequenceAt out of bounds / wrong type
    assert cf._evaluate_node("SequenceAt", [out_seq, np.array([5])], {}) is None

    # SplitToSequence
    data = np.arange(4)
    split = np.array([2])
    out_split = cf._evaluate_node(
        "SplitToSequence", [data, split], {"axis": Attribute("a", "INT", 0)}
    )
    assert len(out_split) == 2
    assert np.all(out_split[0] == [0, 1])

    # SplitToSequence massive
    data_large = np.zeros(10001)
    split_tiny = np.array([1])
    assert (
        cf._evaluate_node(
            "SplitToSequence", [data_large, split_tiny], {"axis": Attribute("a", "INT", 0)}
        )
        is None
    )

    assert (
        cf._evaluate_node("SplitToSequence", [data, None], {"axis": Attribute("a", "INT", 0)})
        is None
    )


def test_cf_gather_scatter_elements():
    cf = ConstantFoldingPass()

    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 0]])
    updates = np.array([[5, 6], [7, 8]])

    out_gather = cf._evaluate_node("GatherElements", [data, indices], {"axis": 0})
    assert out_gather.shape == (2, 2)

    out_scatter = cf._evaluate_node("ScatterElements", [data, indices, updates], {"axis": 0})
    assert out_scatter[0, 0] == 5


def test_cf_shape_size():
    cf = ConstantFoldingPass()
    data = np.zeros((2, 3))

    assert list(cf._evaluate_node("Shape", [data], {})) == [2, 3]
    assert cf._evaluate_node("Size", [data], {}) == 6


def test_cf_nms():
    cf = ConstantFoldingPass()

    boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]]])
    scores = np.array([[[0.9, 0.75]]])
    max_out = np.array([10], dtype=np.int64)
    iou_thresh = np.array([0.5], dtype=np.float32)
    score_thresh = np.array([0.0], dtype=np.float32)

    out_nms = cf._evaluate_node(
        "NonMaxSuppression", [boxes, scores, max_out, iou_thresh, score_thresh], {}
    )
    # returns indices [batch_idx, class_idx, box_idx]
    assert out_nms.shape[0] > 0
    assert out_nms.shape[1] == 3

    # With center point box
    out_nms2 = cf._evaluate_node(
        "NonMaxSuppression",
        [boxes, scores, max_out, iou_thresh, score_thresh],
        {"center_point_box": Attribute("cpb", "INT", 1)},
    )
    assert out_nms2.shape[0] > 0

    # Missing args
    out_nms3 = cf._evaluate_node("NonMaxSuppression", [boxes, scores], {})
    assert out_nms3.shape[0] >= 0


def test_cf_final_lines():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    cf = ConstantFoldingPass()

    # 409: np.any(np.isinf) for array
    t = Tensor("t", (1,), DType.FLOAT32)
    t.data = np.array([float("inf")])
    t.is_initializer = True
    g = Graph("TestInf")
    g.tensors["t"] = t
    n = Node("Identity", ["t"], ["out"])
    g.nodes.append(n)
    cf._run_once(g)  # should hit _has_nan_inf on array with inf

    # 415-418: _has_nan_inf on float and list
    # float:
    t2 = Tensor("t2", (), DType.FLOAT32)
    t2.data = float("inf")
    t2.is_initializer = True
    g.tensors["t2"] = t2
    n2 = Node("Identity", ["t2"], ["out2"])
    g.nodes.append(n2)
    cf._run_once(g)  # float

    # list:
    # Use mock to return list
    t3 = Tensor("t3", (), DType.FLOAT32)
    t3.data = float("nan")
    t3.is_initializer = True
    g.tensors["t3"] = t3

    import pytest

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            ConstantFoldingPass, "_evaluate_node", lambda *args: [np.array([1.0]), float("nan")]
        )
        g.nodes.append(Node("Split", ["t3"], ["out3", "out4"]))
        cf._run_once(g)

    # 725: BN 1D fallback
    Node("BatchNormalization", [], [])
    X = np.ones((2,))
    scale = np.ones((2,))
    B = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))
    out_bn = cf._evaluate_node("BatchNormalization", [X, scale, B, mean, var], {})
    assert out_bn is not None

    # 868: ScatterND reduction without value
    data = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 0]])
    updates = np.array([1.0], dtype=np.float32)
    # Pass 'add' directly not wrapped in Attribute
    out_s = cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": "add"})
    assert out_s[0, 0] == 1.0

    # 913: SplitToSequence without value
    data = np.arange(4)
    split = 2
    out_split = cf._evaluate_node("SplitToSequence", [data, split], {"axis": 0})
    assert len(out_split) == 2


def test_cf_pass_subgraph_recurse_real():
    g = Graph("MainCF_Real")
    sub = Graph("SubCF_Real")

    sub.tensors["X"] = Tensor("X", (2, 3), DType.FLOAT32)
    sub.inputs = [ValueInfo("X", (2, 3), DType.FLOAT32)]
    sub.outputs = ["Y"]
    sub.tensors["Y"] = Tensor("Y", (2,), DType.INT64)

    n_shape = Node("Shape", ["X"], ["Y"])
    sub.nodes.append(n_shape)

    n_if = Node("If", ["cond"], ["Y"])
    n_if.attributes["then"] = Attribute("then", "GRAPH", sub)
    g.nodes.append(n_if)
    g.tensors["cond"] = Tensor("cond", (), DType.BOOL)
    g.tensors["Y"] = Tensor("Y", (), DType.FLOAT32)

    cf = ConstantFoldingPass()
    changed = cf.run(g)
    assert changed


def test_cf_final_lines():
    from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass

    cf = ConstantFoldingPass()

    # 409: np.any(np.isinf) for array
    t = Tensor("t", (1,), DType.FLOAT32)
    t.data = np.array([float("inf")])
    t.is_initializer = True
    g = Graph("TestInf")
    g.tensors["t"] = t
    n = Node("Identity", ["t"], ["out"])
    g.nodes.append(n)
    cf._run_once(g)  # should hit _has_nan_inf on array with inf

    # 415-418: _has_nan_inf on float and list
    # float:
    t2 = Tensor("t2", (), DType.FLOAT32)
    t2.data = float("inf")
    t2.is_initializer = True
    g.tensors["t2"] = t2
    n2 = Node("Identity", ["t2"], ["out2"])
    g.nodes.append(n2)
    cf._run_once(g)  # float

    # list:
    # Use mock to return list
    t3 = Tensor("t3", (), DType.FLOAT32)
    t3.data = float("nan")
    t3.is_initializer = True
    g.tensors["t3"] = t3

    import pytest

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            ConstantFoldingPass, "_evaluate_node", lambda *args: [np.array([1.0]), float("nan")]
        )
        g.nodes.append(Node("Split", ["t3"], ["out3", "out4"]))
        cf._run_once(g)

    # 725: BN 1D fallback
    Node("BatchNormalization", [], [])
    X = np.ones((2,))
    scale = np.ones((2,))
    B = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))
    out_bn = cf._evaluate_node("BatchNormalization", [X, scale, B, mean, var], {})
    assert out_bn is not None

    # 868: ScatterND reduction without value
    data = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 0]])
    updates = np.array([1.0], dtype=np.float32)
    # Pass 'add' directly not wrapped in Attribute
    out_s = cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": "add"})
    assert out_s[0, 0] == 1.0

    # 913: SplitToSequence without value
    data = np.arange(4)
    split = 2
    out_split = cf._evaluate_node("SplitToSequence", [data, split], {"axis": 0})
    assert len(out_split) == 2
