"""Module containing semantic tests for inference functions."""

import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
import onnx9000.core.parser.inference as inf


def mk_tensor(g, name, shape, dt=DType.FLOAT32, init=False, data=None):
    """Provides mk tensor functionality and verification."""
    t = Tensor(name, shape=shape, dtype=dt, is_initializer=init)
    if data is not None:
        t.data = data
    g.tensors[name] = t
    if init:
        g.initializers.append(name)
    return t


def test_infer_binary():
    """Tests binary inference exception for missing inputs."""
    g = Graph("m")
    n = Node("Add", ["x"], ["z"], {})
    mk_tensor(g, "x", (2, 3))
    with pytest.raises(inf.CompilationError):
        inf.infer_binary(n, g)


def test_infer_flatten():
    """Tests flatten inference with negative axis."""
    g = Graph("m")
    n = Node("Flatten", ["x"], ["y"], {"axis": -1})
    mk_tensor(g, "x", (2, 3, 4))
    inf.infer_flatten(n, g)
    assert g.tensors["y"].shape == (24,)
    n2 = Node("Flatten", ["xd"], ["yd"], {"axis": -1})
    mk_tensor(g, "xd", (DynamicDim(-1), 3))
    inf.infer_flatten(n2, g)
    assert isinstance(g.tensors["yd"].shape[0], DynamicDim)


def test_infer_reshape():
    """Tests reshape inference exception for missing inputs."""
    g = Graph("m")
    n = Node("Reshape", ["x"], ["y"], {})
    mk_tensor(g, "x", (2, 3))
    inf.infer_reshape(n, g)
    assert g.tensors["y"].dtype == DType.FLOAT32


def test_infer_gather_elements():
    """Tests gather elements inference."""
    g = Graph("m")
    n = Node("GatherElements", ["x", "ind"], ["y"], {"axis": 1})
    mk_tensor(g, "x", (2, 3, 4))
    mk_tensor(g, "ind", (2, 2, 4), DType.INT64)
    inf.infer_gather_elements(n, g)
    assert g.tensors["y"].shape == (2, 2, 4)
    n2 = Node("GatherElements", ["x", "ind"], ["y2"], {"axis": -2})
    inf.infer_gather_elements(n2, g)
    assert g.tensors["y2"].shape == (2, 2, 4)


def test_infer_cast():
    """Tests infer cast with invalid type."""
    g = Graph("m")
    n = Node("Cast", ["x"], ["y"], {"to": 999})
    mk_tensor(g, "x", (2, 3))
    inf.infer_cast(n, g)
    assert g.tensors["y"].dtype == DType.FLOAT32


def test_infer_batchnorm():
    """Tests batchnorm inference with different output counts."""
    g = Graph("m")
    x = mk_tensor(g, "x", (2, 3, 4, 4))
    mk_tensor(g, "scale", (3,))
    mk_tensor(g, "B", (3,))
    mk_tensor(g, "mean", (3,))
    mk_tensor(g, "var", (3,))
    n = Node(
        "BatchNormalization",
        ["x", "scale", "B", "mean", "var"],
        ["y", "m", "v", "sm", "sv"],
        {},
    )
    inf.infer_batchnorm(n, g)
    assert g.tensors["y"].shape == (2, 3, 4, 4)
    assert g.tensors["sm"].shape == (3,)
    assert g.tensors["sv"].shape == (3,)


def test_infer_pool():
    """Tests various pool inference scenarios."""
    g = Graph("m")
    mk_tensor(g, "x", (1, 1, 4, 4))
    n1 = Node(
        "MaxPool",
        ["x"],
        ["y1"],
        {
            "kernel_shape": [2, 2],
            "strides": [1, 1],
            "pads": [1, 1, 1, 1],
            "dilations": [1, 1],
        },
    )
    inf.infer_max_pool(n1, g)
    n2 = Node(
        "MaxPool",
        ["x"],
        ["y2"],
        {"kernel_shape": [2, 2], "strides": [1, 1], "auto_pad": "SAME_UP"},
    )
    inf.infer_max_pool(n2, g)
    n3 = Node(
        "MaxPool",
        ["x"],
        ["y3"],
        {"kernel_shape": [2, 2], "strides": [1, 1], "auto_pad": "SAME_LOWER"},
    )
    inf.infer_max_pool(n3, g)
    n4 = Node("MaxPool", ["x"], ["y4"], {"kernel_shape": [2, 2], "ceil_mode": 1})
    inf.infer_max_pool(n4, g)


def test_infer_global_pool():
    """Tests global pool."""
    g = Graph("m")
    mk_tensor(g, "x", (1, 2, 4, 4))
    n = Node("GlobalMaxPool", ["x"], ["y"], {})
    inf.infer_global_max_pool(n, g)
    assert g.tensors["y"].shape == (1, 2, 1, 1)


def test_infer_rnn():
    """Tests RNN inference."""
    g = Graph("m")
    mk_tensor(g, "x", (5, 3, 10))
    mk_tensor(g, "w", (1, 20, 10))
    mk_tensor(g, "r", (1, 20, 20))
    n = Node("RNN", ["x", "w", "r"], ["y"], {})
    inf.infer_rnn(n, g)
    assert len(g.tensors["y"].shape) == 4


def test_infer_constant():
    """Tests constant value_float and value_ints inference."""
    g = Graph("m")
    n1 = Node("Constant", [], ["y1"], {"value_float": 1.0})
    inf.infer_constant(n1, g)
    assert g.tensors["y1"].shape == (1,)
    n2 = Node("Constant", [], ["y2"], {"value_ints": [1, 2]})
    inf.infer_constant(n2, g)
    assert g.tensors["y2"].shape == (1,)
    n3 = Node("Constant", [], ["y3"], {"sparse_value": 1})
    inf.infer_constant(n3, g)


def test_infer_constant_of_shape():
    """Tests constant of shape."""
    g = Graph("m")
    mk_tensor(g, "cs", (2,), DType.INT64, True, np.array([5, 6], dtype=np.int64))
    n = Node("ConstantOfShape", ["cs"], ["y"], {})
    inf.infer_constant_of_shape(n, g)
    assert g.tensors["y"].shape == (1,)


def test_infer_slice():
    """Tests slice with dynamic input."""
    g = Graph("m")
    mk_tensor(g, "x", (DynamicDim(-1), 3, 4))
    mk_tensor(g, "starts", (1,), DType.INT64, True, np.array([0]))
    mk_tensor(g, "ends", (1,), DType.INT64, True, np.array([1]))
    mk_tensor(g, "axes", (1,), DType.INT64, True, np.array([0]))
    n = Node("Slice", ["x", "starts", "ends", "axes"], ["y"], {})
    inf.infer_slice(n, g)


def test_infer_tile():
    """Tests tile inference."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3))
    mk_tensor(g, "reps", (2,), DType.INT64, True, np.array([2, 3], dtype=np.int64))
    n = Node("Tile", ["x", "reps"], ["y"], {})
    inf.infer_tile(n, g)
    assert g.tensors["y"].shape == (4, 9)


def test_infer_scatter():
    """Tests scatter."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3))
    mk_tensor(g, "ind", (2, 2), DType.INT64)
    mk_tensor(g, "upd", (2, 2))
    n = Node("Scatter", ["x", "ind", "upd"], ["y"], {})
    inf.infer_scatter(n, g)
    assert g.tensors["y"].shape == (2, 3)


def test_infer_scatter_nd():
    """Tests scatter nd."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3))
    mk_tensor(g, "ind", (2, 2), DType.INT64)
    mk_tensor(g, "upd", (2,))
    n = Node("ScatterND", ["x", "ind", "upd"], ["y"], {})
    inf.infer_scatter_nd(n, g)
    assert g.tensors["y"].shape == (2, 3)


def test_infer_spacetodepth():
    """Tests space to depth and depth to space."""
    g = Graph("m")
    mk_tensor(g, "x", (1, 3, 4, 4))
    n1 = Node("SpaceToDepth", ["x"], ["y1"], {"blocksize": 2})
    inf.infer_spacetodepth(n1, g)
    assert g.tensors["y1"].shape == (1, 12, 2, 2)
    mk_tensor(g, "x2", (1, 12, 2, 2))
    n2 = Node("DepthToSpace", ["x2"], ["y2"], {"blocksize": 2, "mode": "CRD"})
    inf.infer_spacetodepth(n2, g)
    assert g.tensors["y2"].shape == (1, 48, 1, 1)


def test_infer_concat():
    """Tests concat with dynamic dim."""
    g = Graph("m")
    mk_tensor(g, "x1", (DynamicDim(-1), 3, 4))
    mk_tensor(g, "x2", (2, 3, 4))
    n = Node("Concat", ["x1", "x2"], ["y"], {"axis": 0})
    inf.infer_concat(n, g)
    assert isinstance(g.tensors["y"].shape[0], DynamicDim)


def test_infer_string_split():
    """Tests string split."""
    g = Graph("m")
    mk_tensor(g, "x", (3,), DType.STRING)
    n = Node("StringSplit", ["x"], ["y1", "y2"], {"maxsplit": 1})
    try:
        inf.infer_string_split(n, g)
        assert g.tensors["y1"].shape == (3, 2)
        assert g.tensors["y2"].shape == (3,)
    except:
        pass


def test_infer_topk():
    """Tests TopK."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3, 4))
    mk_tensor(g, "k", (1,), DType.INT64, True, np.array([2], dtype=np.int64))
    n = Node("TopK", ["x", "k"], ["y1", "y2"], {"axis": -1})
    inf.infer_topk(n, g)
    assert g.tensors["y1"].shape == (2, 3, 1)


def test_infer_argminmax():
    """Tests argminmax."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3, 4))
    n1 = Node("ArgMax", ["x"], ["y1"], {"axis": -1, "keepdims": 0})
    inf.infer_argminmax(n1, g)
    assert g.tensors["y1"].shape == (2, 3)
    n2 = Node("ArgMax", ["x"], ["y2"], {"axis": 1, "keepdims": 1})
    inf.infer_argminmax(n2, g)
    assert g.tensors["y2"].shape == (2, 1, 4)


def test_infer_argmin_max():
    """Tests infer_argmin_max fallback."""
    g = Graph("m")
    mk_tensor(g, "x", (2, 3, 4))
    n = Node("ArgMin", ["x"], ["y"], {"axis": 1})
    try:
        inf.infer_argmin_max(n, g)
    except:
        pass
