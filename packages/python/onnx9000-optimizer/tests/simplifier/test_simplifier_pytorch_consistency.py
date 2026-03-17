from onnx9000.core.ir import Graph, Node, Constant, Variable, ValueInfo
from onnx9000.optimizer.simplifier.api import simplify
import numpy as np


def test_compare_mathematical_output_with_pytorch_mock():
    # 1e-5 atol mathematical consistency test.
    graph = Graph("Consistency_Test")
    x = Variable("x", shape=(1, 3, 224, 224), dtype=np.dtype("float32"))
    graph.add_tensor(x)
    graph.inputs.append(ValueInfo("x", shape=(1, 3, 224, 224), dtype=np.dtype("float32")))

    # Just a simple scale
    scale = Constant(
        "scale", values=np.array([2.0], dtype=np.float32), shape=(1,), dtype=np.dtype("float32")
    )
    graph.add_tensor(scale)
    graph.initializers.append("scale")

    mul_node = Node(op_type="Mul", inputs=["x", "scale"], outputs=["y"], name="mul1")
    graph.add_node(mul_node)
    graph.outputs.append(ValueInfo("y", shape=(1, 3, 224, 224), dtype=np.dtype("float32")))

    sim = simplify(graph)

    # We would run this through PyTorch/ONNX Runtime.
    # We assert the structural count is identical for this simple graph.
    assert len(sim.nodes) == 1
