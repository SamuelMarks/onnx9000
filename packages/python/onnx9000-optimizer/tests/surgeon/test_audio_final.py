import pytest
from onnx9000.core.ir import Constant, Graph, Node
from onnx9000.optimizer.surgeon.audio import fold_mel_weights


def test_fold_mel_weights_exception():
    graph = Graph("test_graph")

    # Constants for inputs
    c1 = Constant("C1", b"\x00", "int32", [])

    # Mock graph.tensors to crash on setitem
    class CrashingDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.should_crash = False

        def __setitem__(self, key, value):
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
