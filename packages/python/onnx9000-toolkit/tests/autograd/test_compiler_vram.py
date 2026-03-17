from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import track_vram_usage


def test_track_vram():
    g = Graph("test")
    g.add_tensor(Tensor(name="w1", shape=(10, 10), dtype="float32", requires_grad=True))
    g.initializers.append("w1")
    g.add_node(Node("Identity", ["w1"], ["out"]))
    g.outputs.append("out")
    vram = track_vram_usage(g)
    assert vram >= 0.0
