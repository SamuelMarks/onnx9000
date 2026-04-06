import struct

import pytest
from onnx9000.core.ir import Constant, Graph, Node, Variable
from onnx9000.optimizer.surgeon.audio import _unpack_scalar, fold_mel_weights


def test_fold_mel_weights_exception_hit_5_inputs():
    g = Graph("test")
    # We need to pass the `all(isinstance(..., Constant))` check.
    # But then hit the Exception in the body.
    # What throws an exception? Maybe struct.pack with bad values?
    # No, struct.pack("<1f", *weights) where weights are strings?
    # If we make `lower_edge_hertz` very large, maybe it fails math?
    # Or just mock `_unpack_scalar` inside to raise Exception.

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

    from unittest.mock import patch

    with patch("onnx9000.optimizer.surgeon.audio.struct.pack", side_effect=Exception("mock err")):
        fold_mel_weights(g)


def test_fold_mel_weights_sample_rate_zero():
    g = Graph("test")
    bins = Constant("b", shape=(), values=struct.pack("<i", 2))
    dft = Constant("d", shape=(), values=struct.pack("<i", 30))
    sr = Constant("s", shape=(), values=struct.pack("<f", 0.0))  # zero -> becomes 1
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
