import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
import struct

print("Starting debug...")

graph = Graph("test")
graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
w_data = struct.pack(f"<{64 * 27}f", *([1.0] * (64 * 27)))
graph.tensors["W"] = Tensor(
    "W", shape=(64, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
)

graph.nodes.append(Node("Conv", ["X", "W"], ["Y"], name="conv1"))
graph.nodes.append(Node("Relu", ["Y"], ["Z"], name="relu1"))

opt = LayoutOptimizer(graph, False)
opt.inject_transposes()
print("After inject:", [n.op_type for n in graph.nodes])
opt.push_down_transposes()
print("After push down:", [n.op_type for n in graph.nodes])
opt.cancel_transposes()
print("After cancel:", [n.op_type for n in graph.nodes])

print("Finished push down debug.")

graph = Graph("test")
graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), "float32"))
w1_data = struct.pack(f"<{64 * 27}f", *([1.0] * (64 * 27)))
w2_data = struct.pack(f"<{64 * 64 * 9}f", *([1.0] * (64 * 64 * 9)))

graph.tensors["W1"] = Tensor(
    "W1", shape=(64, 3, 3, 3), dtype="float32", is_initializer=True, data=w1_data
)
graph.tensors["W2"] = Tensor(
    "W2", shape=(64, 64, 3, 3), dtype="float32", is_initializer=True, data=w2_data
)

graph.nodes.append(Node("Conv", ["X", "W1"], ["Y"], name="conv1"))
graph.nodes.append(Node("Conv", ["Y", "W2"], ["Z"], name="conv2"))

opt = LayoutOptimizer(graph, False)
opt.inject_transposes()
print("After inject 2:", [n.op_type for n in graph.nodes])
opt.cancel_transposes()
print("After cancel 2:", [n.op_type for n in graph.nodes])
