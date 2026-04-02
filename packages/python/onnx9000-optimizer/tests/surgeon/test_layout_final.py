import pytest
from onnx9000.core.ir import Graph, Node, Attribute, Tensor, Variable
from onnx9000.optimizer.surgeon.layout import optimize_layouts


def test_optimize_layouts_push():
    # Create graph: Input -> Transpose(NCHW->NHWC) -> Relu -> Output
    graph = Graph("test_graph")
    t1 = Variable("X", [1, 3, 224, 224], "float32")
    t2 = Variable("T1", [1, 224, 224, 3], "float32")
    t3 = Variable("Y", [1, 224, 224, 3], "float32")

    n1 = Node(
        "Transpose", ["X"], ["T1"], {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}, "trans"
    )
    n2 = Node("Relu", ["T1"], ["Y"], {}, "relu")

    # Setup graph relations manually
    t1.outputs.append(n1)
    n1.inputs = ["X"]
    n1.outputs = ["T1"]

    t2.inputs.append(n1)
    t2.outputs.append(n2)
    n2.inputs = ["T1"]
    n2.outputs = ["Y"]

    graph.nodes = [n1, n2]
    graph.tensors = {"X": t1, "T1": t2, "Y": t3}

    optimized = optimize_layouts(graph)

    # Should have Relu -> Transpose
    assert optimized.nodes[0].op_type == "Relu"
    assert optimized.nodes[1].op_type == "Transpose"
    assert optimized.nodes[0].inputs[0] == "X"
