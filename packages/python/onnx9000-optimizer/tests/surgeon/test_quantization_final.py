import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Constant
from onnx9000.optimizer.surgeon.quantization import quantize_ptq


def test_quantize_ptq_basic():
    graph = Graph("test_graph")

    # Weight constant
    w_data = np.random.rand(3, 3, 3, 32).astype(np.float32)
    w_const = Constant("W", w_data.tobytes(), "float32", [3, 3, 3, 32])
    graph.tensors["W"] = w_const
    graph.initializers.append("W")

    n1 = Node("Conv", ["X", "W"], ["Y"], {}, "conv")
    graph.nodes = [n1]

    quantized = quantize_ptq(graph)

    assert graph.tensors["W"].dtype == "uint8"
    assert "W_quantized" in graph.metadata
    assert float(graph.metadata["W_scale"]) > 0
