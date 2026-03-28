"""Tests for the EdgeTPU optimizer in the TFLite exporter."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.optimizations.edgetpu import EdgeTPUOptimizer


def test_edgetpu_optimizer():
    """Test various EdgeTPU-specific optimizations and rewrites."""
    graph = Graph("test")

    # 1D Convolution
    graph.tensors["X"] = Tensor("X", shape=(1, 3, 224), dtype="float32", is_initializer=False)
    graph.inputs.append(ValueInfo("X", (1, 3, 224), "float32"))
    graph.nodes.append(Node("Conv", ["X"], ["Y"], name="conv1d"))

    # MatMul
    graph.nodes.append(Node("MatMul", ["A", "B"], ["C"], name="matmul1"))

    # LeakyRelu
    graph.nodes.append(Node("LeakyRelu", ["X"], ["L"], name="leaky1"))

    # Dynamic Strided Slice
    graph.nodes.append(
        Node("Slice", ["X", "Starts", "Ends", "Axes", "Steps"], ["S"], name="slice1")
    )

    # Softmax
    graph.nodes.append(Node("Softmax", ["X"], ["S2"], name="soft1"))

    # Loop
    graph.nodes.append(Node("Loop", ["X"], ["L1"], name="loop1"))

    # Padding
    graph.tensors["W_conv"] = Tensor(
        "W_conv", shape=(16, 5, 3, 3), dtype="float32", is_initializer=True, data=b""
    )
    graph.nodes.append(Node("Conv", ["X2", "W_conv"], ["Y2"], name="conv2"))

    optimizer = EdgeTPUOptimizer(graph)
    warnings = optimizer.optimize()

    assert any("Replaced 1 1D Convolutions" in w for w in warnings)
    assert any("Expanded 1 MatMul" in w for w in warnings)
    assert any("Emulated 1 LeakyRelu" in w for w in warnings)
    assert any("Dynamic StridedSlice detected" in w for w in warnings)
    assert any("Injected Zero-Padding into 1 Convolutions" in w for w in warnings)
    assert any("Operation Loop (loop1) breaks strict NNAPI" in w for w in warnings)
    assert any("Rewrote 1 Softmax operations" in w for w in warnings)
