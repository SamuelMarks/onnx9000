"""Tests for audio final."""

import pytest
from onnx9000.core.ir import Constant, Graph, Node
from onnx9000.optimizer.surgeon.audio import fold_mel_weights


def test_fold_mel_weights_exception():
    """Docstring for D103."""
    graph = Graph("test_graph")

    # Constants for inputs
    c1 = Constant("C1", b"\x00", "int32", [])

    # Mock graph.tensors to crash on setitem
    class CrashingDict(dict):
        """Crashing dict."""

        def __init__(self, *args, **kwargs):
            """Init."""
            super().__init__(*args, **kwargs)
            self.should_crash = False

        def __setitem__(self, key, value):
            """Setitem."""
            if self.should_crash:
                raise Exception("force crash")
            super().__setitem__(key, value)

    graph.tensors = CrashingDict()
    graph.tensors["C1"] = c1

    n1 = Node("MelWeightMatrix", ["C1"], ["OUT"], {}, "mel")
    graph.nodes = [n1]

    # Enable crash
    graph.tensors.should_crash = True

    fold_mel_weights(graph)
    # If we reached here, it continued past the exception
    assert len(graph.nodes) == 1


def test_fold_mel_weights_success():
    """Test successful generation of Mel weights."""
    import struct

    from onnx9000.core.dtypes import DType

    graph = Graph("test_graph")

    # Inputs: num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz
    c1 = Constant("C1", struct.pack("<i", 80), shape=[], dtype=DType.INT32)
    c2 = Constant("C2", struct.pack("<i", 1024), shape=[], dtype=DType.INT32)
    c3 = Constant("C3", struct.pack("<i", 16000), shape=[], dtype=DType.INT32)
    c4 = Constant("C4", struct.pack("<f", 0.0), shape=[], dtype=DType.FLOAT32)
    c5 = Constant("C5", struct.pack("<f", 8000.0), shape=[], dtype=DType.FLOAT32)

    graph.add_tensor(c1)
    graph.add_tensor(c2)
    graph.add_tensor(c3)
    graph.add_tensor(c4)
    graph.add_tensor(c5)

    n1 = Node("MelWeightMatrix", ["C1", "C2", "C3", "C4", "C5"], ["OUT"], {}, "mel")
    graph.add_node(n1)

    fold_mel_weights(graph)

    assert len(graph.nodes) == 0
    assert "OUT" in graph.tensors
    out_c = graph.tensors["OUT"]
    assert out_c.shape == (513, 80)
    assert out_c.dtype == DType.FLOAT32
