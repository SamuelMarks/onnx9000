from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.shape_inference import infer_shapes_and_types


def test_infer_shapes_and_types_add() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")
    g.inputs.append("B")
    n = Node("Add", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 20)
    assert g.tensors["Y"].dtype == DType.FLOAT32


def test_infer_shapes_and_types_matmul() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(20, 30), dtype=DType.FLOAT32))
    g.inputs.append("A")
    g.inputs.append("B")
    n = Node("MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 30)


def test_infer_shapes_and_types_relu() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.inputs.append("A")
    n = Node("Relu", inputs=["A"], outputs=["Y"], attributes={})
    g.add_node(n)
    infer_shapes_and_types(g)
    assert "Y" in g.tensors
    assert g.tensors["Y"].shape == (10, 20)
