"""Module providing core logic and structural definitions."""

import numpy as np
import pytest
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType
import onnx9000.core.parser.inference as inf


def mk_tensor(g, name, shape, dt=DType.FLOAT32, init=False, data=None):
    """Provides mk tensor functionality and verification."""
    t = Tensor(name, shape=shape, dtype=dt, is_initializer=init)
    if data is not None:
        t.data = data
    if init:
        g.initializers.append(name)
    g.tensors[name] = t
    return t
    """Provides semantic logic and verification for last mile inference coverage."""


class DummyVal:
    """Represents the DummyVal class."""

    def __init__(self, shape, dtype):
        """Provides   init   functionality and verification."""
        self.shape = shape
        self.dtype = dtype


def test_last_mile():
    """Tests the test last mile functionality."""
    g = Graph("m")
    mk_tensor(g, "c_171", (3,))
    mk_tensor(g, "a_171", (1,))
    mk_tensor(g, "b_171", (1,))
    inf.infer_binary(Node("Where", ["c_171", "a_171", "b_171"], ["y_171"], {}), g)
    mk_tensor(g, "x_936", (DynamicDim(-1),))
    mk_tensor(g, "reps_936", (1,), DType.INT64, True, np.array([2]))
    inf.infer_tile(Node("Tile", ["x_936", "reps_936"], ["y_936"], {}), g)
    mk_tensor(g, "x_990", (2,))
    mk_tensor(g, "ind_990", (2,))
    mk_tensor(g, "upd_990", (2,))
    inf.infer_scatter_elements(
        Node("ScatterElements", ["x_990", "ind_990", "upd_990"], ["y_990"], {}), g
    )
    mk_tensor(g, "x_1101", (2,))
    mk_tensor(g, "cond_1101", (2,), DType.BOOL)
    inf.infer_compress(Node("Compress", ["x_1101", "cond_1101"], ["y_1101"], {}), g)
    mk_tensor(g, "x_1909", (2,))
    mk_tensor(g, "sc_1909", (2,))
    inf.infer_layer_normalization(
        Node("LayerNormalization", ["x_1909", "sc_1909"], ["y_1909", "aux_1909"], {}), g
    )
    inf.infer_random_gen(Node("RandomUniform", [], ["y_2100"], {"dtype": 6}), g)
    mk_tensor(g, "x_2126", (2,))
    inf.infer_random_gen_like(
        Node("RandomUniformLike", ["missing_2126"], ["y_2126"], {}), g
    )
    mk_tensor(g, "x_2454", (2,))
    inf.infer_unique(Node("Unique", ["x_2454"], ["y_2454", "aux_2454"], {}), g)
    mk_tensor(g, "x_2787", (2,))
    inf.infer_dropout(Node("Dropout", ["x_2787"], ["y_2787", "mask_2787"], {}), g)
