"""Module docstring."""

from onnx9000.autograd.shape_inference import infer_backward_shapes
from onnx9000.ir import Graph, Node
from onnx9000.dtypes import DType
import onnx9000


def test_shape_inference():
    """test_shape_inference docstring."""
    graph = Graph("test")
    # Original tensor
    graph.tensors["x"] = onnx9000.Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")

    # Backward node
    bwd_node = Node(
        "ReluGrad", ["grad_y", "x"], ["grad_x_wrt_relu"], {}, name="relu_bwd"
    )
    graph.add_node(bwd_node)

    infer_backward_shapes(graph)

    assert "grad_x_wrt_relu" in graph.tensors
    assert graph.tensors["grad_x_wrt_relu"].shape == (10, 20)
    assert graph.tensors["grad_x_wrt_relu"].dtype == DType.FLOAT32
