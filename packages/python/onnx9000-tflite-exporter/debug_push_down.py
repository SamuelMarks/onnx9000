from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
import struct

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
print("After inject:")
for n in graph.nodes:
    print(n.op_type, n.inputs, n.outputs)

print("Running push down...")
opt.push_down_transposes()
print("Done push down")
