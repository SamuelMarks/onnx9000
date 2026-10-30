"""Unit tests for shape and type inference in `onnx9000.core.shape_inference`."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, ValueInfo, Variable
from onnx9000.core.shape_inference import _promote_types, get_attr, infer_shapes_and_types


def test_shape_inference_functions() -> None:
    """Verify _promote_types, get_attr, and infer_shapes_and_types."""
    assert _promote_types(DType.FLOAT16, DType.FLOAT32) == DType.FLOAT32

    node = Node("Relu", ["x"], ["y"], attributes={"alpha": Attribute("alpha", "FLOAT", 0.1)})
    assert get_attr(node, "alpha") == 0.1

    g = Graph("si_test")
    g.inputs.append(ValueInfo("x", (2, 4), DType.FLOAT32))
    g.add_tensor(Variable("x", (2, 4), DType.FLOAT32))
    g.add_node(node)
    infer_shapes_and_types(g)
    assert g.tensors["y"].shape == (2, 4)
    assert g.tensors["y"].dtype == DType.FLOAT32
