import pytest
import struct
from onnx9000.core.ir import Constant, Graph, Node, Variable
from onnx9000.optimizer.surgeon.audio import _unpack_scalar, fold_mel_weights


def test_fold_mel_weights_exception_hit_5_inputs():
    g = Graph("test")
    n = Node("MelWeightMatrix", inputs=["i1", "i2", "i3", "i4", "i5"], outputs=["out"])
    g.add_node(n)
    fold_mel_weights(g)


def test_fold_mel_weights_upper_bound_hit_third_try():
    g = Graph("test")
    bins = Constant("b", shape=(), values=struct.pack("<i", 2))
    dft = Constant("d", shape=(), values=struct.pack("<i", 30))
    sr = Constant("s", shape=(), values=struct.pack("<f", 1000.0))
    low = Constant("l", shape=(), values=struct.pack("<f", 0.0))
    high = Constant("h", shape=(), values=struct.pack("<f", 500.0))
    g.add_tensor(bins)
    g.add_tensor(dft)
    g.add_tensor(sr)
    g.add_tensor(low)
    g.add_tensor(high)
    out = Variable("out")
    g.add_tensor(out)
    n = Node("MelWeightMatrix", inputs=["b", "d", "s", "l", "h"], outputs=["out"])
    g.add_node(n)

    fold_mel_weights(g)


def test_unpack_scalar_int_float():
    c = Constant("c", shape=(), values=0)
    c.data = 3.14
    assert _unpack_scalar(c) == 3.14


def test_unpack_scalar_8bytes():
    c = Constant("c", shape=(), values=0)
    c.data = struct.pack("<d", 3.14)
    assert abs(_unpack_scalar(c, True) - 3.14) < 1e-6

    c2 = Constant("c2", shape=(), values=0)
    c2.data = struct.pack("<q", 42)
    assert _unpack_scalar(c2, False) == 42.0


def test_unpack_scalar_none():
    c = Constant("c", shape=(), values=None)
    assert _unpack_scalar(c) == 0.0


def test_unpack_scalar_fallback():
    c = Constant("c", shape=(), values=0)
    c.data = b"toolong"
    assert _unpack_scalar(c) == 0.0
