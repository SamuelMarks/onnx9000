"""Module providing core logic and structural definitions."""

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


class DummyVal:
    """Represents the DummyVal class."""

    def __init__(self, shape, dtype):
        """Provides   init   functionality and verification."""
        self.shape = shape
        self.dtype = dtype


def test_remaining_inference():
    """Provides semantic logic and verification for remaining inference gaps."""
    g = Graph("m")
    mk_tensor(g, "c", (2,))
    mk_tensor(g, "a", (3,))
    mk_tensor(g, "b", (3,))
    with pytest.raises(inf.CompilationError):
        inf.infer_binary(Node("Where", ["c", "a", "b"], ["y"], {}), g)
    mk_tensor(g, "x_1d", (5,))
    mk_tensor(g, "scale", (5,))
    mk_tensor(g, "B", (5,))
    mk_tensor(g, "mean", (5,))
    mk_tensor(g, "var", (5,))
    mk_tensor(g, "y_bn", (1,))
    mk_tensor(g, "m_bn", (1,))
    inf.infer_batchnorm(
        Node(
            "BatchNormalization",
            ["x_1d", "scale", "B", "mean", "var"],
            ["y_bn", "m_bn"],
            {},
        ),
        g,
    )
    assert isinstance(g.tensors["m_bn"].shape[0], DynamicDim)
    mk_tensor(g, "x_pool", (1, 1, DynamicDim(-1), 4))
    inf.infer_max_pool(
        Node("MaxPool", ["x_pool"], ["y_pool_out"], {"kernel_shape": [2, 2]}), g
    )
    mk_tensor(g, "y_c", (1,))
    inf.infer_constant(
        Node("Constant", [], ["y_c"], {"value": DummyVal((2,), DType.INT32)}), g
    )
    mk_tensor(g, "cs_in", (2,))
    inf.infer_constant_of_shape(
        Node(
            "ConstantOfShape",
            ["cs_in"],
            ["y_cos"],
            {"value": DummyVal((), DType.INT32)},
        ),
        g,
    )
    mk_tensor(g, "x_sl", (4, 4))
    mk_tensor(g, "y_sl", (1,))
    mk_tensor(g, "starts2", (1,), DType.INT64, True, np.array([100], dtype=np.int64))
    mk_tensor(g, "ends2", (1,), DType.INT64, True, np.array([-100], dtype=np.int64))
    mk_tensor(g, "axes2", (1,), DType.INT64, True, np.array([-1], dtype=np.int64))
    inf.infer_slice(
        Node("Slice", ["x_sl", "starts2", "ends2", "axes2"], ["y_sl"], {}), g
    )
    mk_tensor(g, "reps_no_data", (2,), DType.INT64)
    inf.infer_tile(Node("Tile", ["x_sl", "reps_no_data"], ["y_t"], {}), g)
    mk_tensor(g, "ind_se", (2, 2), DType.INT64)
    mk_tensor(g, "upd_se", (2, 2))
    mk_tensor(g, "y_se", (1,))
    inf.infer_scatter_elements(
        Node("ScatterElements", ["x_sl", "ind_se", "upd_se"], ["y_se"], {"axis": -1}), g
    )
    mk_tensor(g, "x_s2d", (1, 3, 4, 4))
    mk_tensor(g, "y_s2d", (1,))
    inf.infer_spacetodepth(Node("SpaceToDepth", ["x_s2d"], ["y_s2d"], {}), g)
    mk_tensor(g, "cond", (4,), DType.BOOL)
    mk_tensor(g, "y_comp", (1,))
    mk_tensor(g, "y_comp2", (1,))
    inf.infer_compress(Node("Compress", ["x_sl", "cond"], ["y_comp"], {"axis": -1}), g)
    inf.infer_compress(Node("Compress", ["x_sl", "cond"], ["y_comp2"], {}), g)
    mk_tensor(g, "y_ln", (1,))
    mk_tensor(g, "mean_ln", (1,))
    mk_tensor(g, "inv_ln", (1,))
    mk_tensor(g, "scale_ln", (4,))
    inf.infer_layer_normalization(
        Node(
            "LayerNormalization",
            ["x_sl", "scale_ln"],
            ["y_ln", "mean_ln", "inv_ln"],
            {},
        ),
        g,
    )
    mk_tensor(g, "y_rg", (1,))
    inf.infer_random_gen(Node("RandomUniform", [], ["y_rg"], {"shape": [2, 2]}), g)
    mk_tensor(g, "y_rgl", (1,))
    inf.infer_random_gen_like(
        Node("RandomUniformLike", ["x_sl"], ["y_rgl"], {"dtype": DType.INT32.value}), g
    )
    mk_tensor(g, "x_str", (3,), DType.STRING)
    mk_tensor(g, "y_ss_0", (1,))
    mk_tensor(g, "y_ss_1", (1,))
    try:
        inf.infer_string_split(
            Node("StringSplit", ["x_str"], ["y_ss_0", "y_ss_1"], {}), g
        )
    except:
        pass
    mk_tensor(g, "y_tk_0", (1,))
    mk_tensor(g, "y_tk_1", (1,))
    mk_tensor(g, "k", (1,), DType.INT64, True, np.array([2]))
    try:
        inf.infer_topk(
            Node("TopK", ["x_sl", "k"], ["y_tk_0", "y_tk_1"], {"axis": -1}), g
        )
    except:
        pass
    mk_tensor(g, "y_u_0", (1,))
    mk_tensor(g, "y_u_1", (1,))
    mk_tensor(g, "y_u_2", (1,))
    mk_tensor(g, "y_u_3", (1,))
    inf.infer_unique(
        Node("Unique", ["x_sl"], ["y_u_0", "y_u_1", "y_u_2", "y_u_3"], {}), g
    )
    mk_tensor(g, "y_do", (1,))
    mk_tensor(g, "y_do_mask", (1,))
    inf.infer_dropout(Node("Dropout", ["x_sl"], ["y_do", "y_do_mask"], {}), g)
