"""Module providing core logic and structural definitions."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.shape_inference import infer_backward_shapes


def test_unbroadcasting_shape_preservation() -> None:
    """Test VJP un-broadcasting safely handles implicit right-alignment natively."""
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    # Forward: [2, 3, 4] + [4] (implicitly aligned right)
    graph = Graph("test")
    graph.tensors["x1"] = Tensor(shape=(2, 3, 4), dtype=DType.FLOAT32, name="x1")
    graph.tensors["x2"] = Tensor(shape=(4,), dtype=DType.FLOAT32, name="x2")
    fwd_node = Node("Add", ["x1", "x2"], ["y"], {}, name="add")

    rule = get_vjp_rule("Add")
    nodes, grads = rule.build_backward_nodes(fwd_node, ["grad_y"])

    # We just ensure the structural un-broadcast nodes (ReduceSum) were added accurately.
    assert len(nodes) > 1
    assert any(n.op_type == "ReduceSum" for n in nodes)
    # Target shape tensor logic should be properly referenced
    assert any(n.op_type == "Shape" and n.inputs[0] == "x2" for n in nodes)


def test_shape_inference() -> None:
    """Tests the test shape inference functionality."""
    graph = Graph("test")
    graph.tensors["x"] = Tensor(shape=(10, 20), dtype=DType.FLOAT32, name="x")
    bwd_node = Node("ReluGrad", ["grad_y", "x"], ["grad_x_wrt_relu"], {}, name="relu_bwd")
    graph.add_node(bwd_node)
    infer_backward_shapes(graph)
    assert "grad_x_wrt_relu" in graph.tensors
    assert graph.tensors["grad_x_wrt_relu"].shape == (10, 20)
    assert graph.tensors["grad_x_wrt_relu"].dtype == DType.FLOAT32
